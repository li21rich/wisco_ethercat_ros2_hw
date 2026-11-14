#!/usr/bin/env python3
"""Extract arm01's link meshes, joint frames and placeholder inertials from the STEP.

Scaffolding for urdf/arm01/arm01.urdf (the URDF is the source of truth; this
script only computes numbers and writes meshes). Needs OCP (cadquery-ocp).

    python tools/cad/extract_arm01.py "Full Assembly (2).step" \
        --meshes urdf/arm01/meshes --json tools/cad/arm01_extract.json

What it does, in CAD millimetres, then converted to the base frame in metres:
  1. reads every leaf solid with its world placement; drops duplicate
     instances (the export repeats J2's actuator at top level),
  2. groups actuator solids into the 7 instances and fits each joint axis
     from the actuator's large cylindrical faces (area-weighted),
  3. assigns every solid to a link (table below; fasteners and anything not
     listed go to the nearest assigned part),
  4. rotates everything past the elbow by -theta about the J4 axis so the
     URDF zero is a straight elbow (the CAD has the elbow flexed),
  5. writes one binary STL per link in that link's frame, and computes
     mass / COM / inertia per link from placeholder densities.

Base frame: origin at the top centre of the post (CAD (0, 0, 541) mm),
x = forward = CAD -y, y = CAD +x, z = up. Every link frame is parallel to the
base frame at the zero pose; joint axes are given as vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.GProp import GProp_GProps
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.gp import gp_Trsf

# ----------------------------------------------------------------------------- constants
ORIGIN_CAD = np.array([0.0, 0.0, 541.0])          # top centre of the post, mm
R_CB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)   # CAD -> base rotation

# densities, kg/mm^3 -- PLACEHOLDERS until link masses are weighed
RHO_ACTUATOR = 4.5e-6       # mixed steel/copper/aluminium, gives ~0.4 kg per X4
RHO_STRUCTURE = 2.7e-6      # aluminium
RHO_FASTENER = 7.85e-6      # steel

ACTUATOR_MODELS = {'X6-P20-60': 'x6_60', 'X4-P36-36': 'x4_36', 'X4-P12.5-10': 'x4_10'}
# rough instance centres (CAD mm) to sort actuator solids into joints
SEEDS = {
    'J1': ('X6-P20-60', (-162.8, 0.2, 492.6)),
    'J2': ('X6-P20-60', (-249.7, 5.3, 541.9)),
    'J3': ('X4-P36-36', (-248.8, -13.2, 447.3)),
    'J4': ('X4-P36-36', (-246.2, -53.4, 358.2)),
    'J5': ('X4-P36-36', (-244.0, -63.3, 291.4)),
    'J6': ('X4-P12.5-10', (-242.3, -100.0, 233.8)),
    'J7': ('X4-P12.5-10', (-235.2, -139.9, 175.1)),
}
# solids on each actuator's output face, by volume (mm^3); the rest is housing
OUTPUT_VOLUMES = {'X6-P20-60': (10458, 6024), 'X4-P36-36': (3728, 2542),
                  'X4-P12.5-10': (6861, 4021, 1635)}
# which link holds the housing and which the output, per joint. J3 is mounted
# with its output on the shoulder side and its housing on the upper arm.
ACTUATOR_LINKS = {
    'J1': ('base_link', 'link1'),
    'J2': ('link1', 'link2'),
    'J3': ('link3', 'link2'),
    'J4': ('link3', 'link4'),
    'J5': ('link4', 'link5'),
    'J6': ('link5', 'wrist_crank'),     # J6 drives the wrist through a parallelogram
    'J7': ('link6', 'link7'),
}
# structural parts by (subassembly, name)
PARTS = {
    'base_link': [('40-2540', '*'), ('left30degree', '*'), ('', 'Extrude1'),
                  ('', '40-240009__M10x1.5 Tapped Hole1')],
    'link1': [('Left Shoulder', n) for n in ('J1 Hardstop Spacer', 'J1-J2 Middle Link', 'J1-J2 Back Link',
                                             'J1-J2 Front Link', 'J2 Bearing Block')],
    'link2': [('Left Shoulder', n) for n in ('J2 Hardstop Spacer', 'J2-J3 Back Linkage', 'J2-J3 Middle Link',
                                             'J2-J3 Front Link', 'Part 2', 'J3 Hardstop Spacer')],
    'link3': [('elbow v4 Left', n) for n in ('Spacer so bend doesnt intersect', 'J3-J4 Back Linkage',
                                             'J3-J4 Front Linkage', 'Top Elbow Standoff', 'Elbow Bearing Block')],
    'link4': [('elbow v4 Left', n) for n in ('J4-J5 Back Linkage', 'J4-J5 Front Linkage', 'Bottom Elbow Standoff',
                                             'Part 2', 'Elbow Hardstop Spacer')],
    'link5': [('Wrist v3 LEFT', n) for n in ('Elbow Hardstop Spacer', 'Wrist Top Plate', 'Wrist Bottom Plate',
                                             'Part 15', 'Part 22')],
    'wrist_crank': [('Wrist v3 LEFT', 'Part 18')],
    'wrist_rod': [('Wrist v3 LEFT', n) for n in ('Part 19', 'Part 2', 'Cut-Sweep1')],
    'link6': [('Wrist v3 LEFT', n) for n in ('Part 14', 'Part 16', 'Part 17', 'Part 20')],
    'link7': [('Wrist v3 LEFT', n) for n in ('Part 23', 'Hand Bracket')] + [('', 'J7PEG')],
}
FASTENER_WORDS = ('screw', 'washer', 'nut', 'dowel', 'insert', 'bolt', 'pin')
DOWNSTREAM_OF_ELBOW = ('link4', 'link5', 'wrist_crank', 'wrist_rod', 'link6', 'link7')


# ----------------------------------------------------------------------------- STEP read
def read_leaves(path):
    doc = TDocStd_Document(TCollection_ExtendedString('doc'))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(path) != 1:
        raise SystemExit('cannot read %s' % path)
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def name(label):
        a = TDataStd_Name()
        return a.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), a) else '?'

    out = []

    def walk(label, trsf, path):
        ref = label
        t = gp_Trsf()
        t.Multiply(trsf)
        if st.IsReference_s(label):
            ref = TDF_Label()
            st.GetReferredShape_s(label, ref)
            t.Multiply(st.GetLocation_s(label).Transformation())
        comps = TDF_LabelSequence()
        st.GetComponents_s(ref, comps)
        if comps.Length():
            for i in range(1, comps.Length() + 1):
                walk(comps.Value(i), t, path + [name(label)])
            return
        shape = st.GetShape_s(ref).Moved(TopLoc_Location(t))
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, g)
        c = g.CentreOfMass()
        M = g.MatrixOfInertia()
        inertia = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        sub = [p.replace(' <1>', '') for p in path[1:]]
        out.append(dict(name=name(label), sub=sub[-1] if sub else '', path='/'.join(sub), shape=shape,
                        vol=g.Mass(), com=np.array([c.X(), c.Y(), c.Z()]), inertia=inertia,
                        bbox=np.array(box.Get()) if not box.IsVoid() else None))

    free = TDF_LabelSequence()
    st.GetFreeShapes(free)
    for i in range(1, free.Length() + 1):
        walk(free.Value(i), gp_Trsf(), [])
    # duplicate instances: same name, centroid within 1 mm, volume within 1%
    keep = []
    by_name = {}
    for leaf in out:
        dup = False
        for other in by_name.get(leaf['name'], []):
            if (np.linalg.norm(other['com'] - leaf['com']) < 1.0
                    and abs(other['vol'] - leaf['vol']) <= 0.01 * max(abs(other['vol']), 1.0)):
                dup = True
                break
        if not dup:
            by_name.setdefault(leaf['name'], []).append(leaf)
            keep.append(leaf)
    return keep, len(out)


# ----------------------------------------------------------------------------- geometry helpers
def cylinder_axis(leaves, min_radius=15.0):
    M = np.zeros((3, 3))
    cyl = []
    for leaf in leaves:
        ex = TopExp_Explorer(leaf['shape'], TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face_s(ex.Current())
            s = BRepAdaptor_Surface(f)
            if s.GetType() == GeomAbs_Cylinder and s.Cylinder().Radius() > min_radius:
                ax = s.Cylinder().Axis()
                d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
                p = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
                g = GProp_GProps()
                BRepGProp.SurfaceProperties_s(f, g)
                M += g.Mass() * np.outer(d, d)
                cyl.append((p, d, g.Mass()))
            ex.Next()
    w, v = np.linalg.eigh(M)
    axis = v[:, -1]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d, a in cyl:
        if abs(d @ axis) > 0.99:
            P = np.eye(3) - np.outer(axis, axis)
            A += a * P
            b += a * P @ p
    point = np.linalg.lstsq(A, b, rcond=None)[0]
    return axis, point, float(w[-2] / w[-1])


def closest_point_on_line(p, d, q):
    return p + d * ((q - p) @ d)


def closest_between_lines(p1, d1, p2, d2):
    """Midpoint of the shortest segment between two lines, and its length."""
    n = np.cross(d1, d2)
    A = np.array([d1, -d2, n]).T
    t = np.linalg.solve(A, p2 - p1)
    a = p1 + t[0] * d1
    b = p2 + t[1] * d2
    return 0.5 * (a + b), float(np.linalg.norm(a - b))


def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def signed_angle(u, v, axis):
    pu = u - axis * (u @ axis)
    pv = v - axis * (v @ axis)
    return math.atan2(axis @ np.cross(pu, pv), pu @ pv)


def frame_rpy(z, x_hint):
    """URDF rpy (fixed XYZ) of a frame with z along `z` and x as close to `x_hint` as possible."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - z * (np.asarray(x_hint, float) @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [roll, pitch, yaw]


def point_box_distance(p, box):
    lo, hi = box[:3], box[3:]
    d = np.maximum(np.maximum(lo - p, 0), p - hi)
    return float(np.linalg.norm(d))


def triangles(shape, tol=0.1):
    BRepMesh_IncrementalMesh(shape, tol, False, 0.35, True)
    tris = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            T = loc.Transformation()
            nodes = []
            for i in range(1, tri.NbNodes() + 1):
                q = tri.Node(i).Transformed(T)
                nodes.append((q.X(), q.Y(), q.Z()))
            nodes = np.array(nodes)
            rev = face.Orientation() == 1   # TopAbs_REVERSED
            idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            if rev:
                idx = idx[:, [0, 2, 1]]
            tris.append(nodes[idx])
        ex.Next()
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def write_stl(path, tris):
    tris = np.asarray(tris, np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 0, n / np.where(ln > 0, ln, 1), 0).astype(np.float32)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n
    rec['v'] = tris
    with open(path, 'wb') as f:
        f.write(b'arm01 link mesh, metres, link frame'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step')
    ap.add_argument('--meshes', default='urdf/arm01/meshes')
    ap.add_argument('--json', default='tools/cad/arm01_extract.json')
    ap.add_argument('--tol', type=float, default=0.15, help='tessellation tolerance, mm')
    args = ap.parse_args()

    leaves, total = read_leaves(args.step)
    print('%d leaves, %d after dropping duplicate instances' % (total, len(leaves)), file=sys.stderr)

    # -- actuators
    inst = {j: [] for j in SEEDS}
    for leaf in leaves:
        if leaf['name'] in ACTUATOR_MODELS:
            cands = [j for j, (m, _) in SEEDS.items() if m == leaf['name']]
            j = min(cands, key=lambda j: np.linalg.norm(leaf['com'] - np.array(SEEDS[j][1])))
            if np.linalg.norm(leaf['com'] - np.array(SEEDS[j][1])) > 60.0:
                continue        # stray zero-volume shells; nearest-part rule below
            inst[j].append(leaf)
            leaf['joint'] = j
    joints = {}
    for j, members in inst.items():
        axis, point, ratio = cylinder_axis(members)
        vol = sum(abs(m['vol']) for m in members)
        com = sum(abs(m['vol']) * m['com'] for m in members) / vol
        center = closest_point_on_line(point, axis, com)
        joints[j] = dict(axis=axis, center=center, eig_ratio=ratio, solids=len(members))
        model = SEEDS[j][0]
        house, out = ACTUATOR_LINKS[j]
        # big solids by volume; small ones (screws, shells) by where they sit
        # along the axis: past 60% of the way from the body to the output face
        big_out = [m for m in members if any(abs(abs(m['vol']) - v) < 2.0 for v in OUTPUT_VOLUMES[model])]
        body = max(members, key=lambda m: abs(m['vol']))
        t_body = (body['com'] - center) @ axis
        t_out = np.mean([(m['com'] - center) @ axis for m in big_out])
        t_split = t_body + 0.6 * (t_out - t_body)
        for m in members:
            if m in big_out:
                is_out = True
            elif abs(m['vol']) >= 1000:
                is_out = False
            else:
                t = (m['com'] - center) @ axis
                is_out = (t - t_split) * np.sign(t_out - t_body) > 0
            m['link'] = out if is_out else house
            m['rho'] = RHO_ACTUATOR

    # axis signs (see the URDF ledger): positive = lift forward (J1), abduct (J2),
    # yaw about the upward arm axis (J3, J5), flex (J4, J6), and about the
    # outward hand axis (J7)
    def orient(v, ref):
        return v if v @ np.array(ref, float) > 0 else -v

    joints['J1']['axis'] = orient(joints['J1']['axis'], (-1, 0, 0))
    joints['J2']['axis'] = orient(joints['J2']['axis'], (0, 1, 0))
    joints['J3']['axis'] = orient(joints['J3']['axis'], (0, 0, 1))
    joints['J4']['axis'] = orient(joints['J4']['axis'], (-1, 0, 0))
    joints['J5']['axis'] = orient(joints['J5']['axis'], (0, 0, 1))
    joints['J6']['axis'] = orient(joints['J6']['axis'], (-1, 0, 0))
    joints['J7']['axis'] = orient(joints['J7']['axis'], (0, -1, 0))

    # -- structural parts
    table = {}
    for link, entries in PARTS.items():
        for sub, nm in entries:
            table.setdefault((sub, nm), link)
    unassigned = []
    for leaf in leaves:
        if 'link' in leaf:
            continue
        low = leaf['name'].lower()
        fast = any(w in low for w in FASTENER_WORDS)
        leaf['rho'] = RHO_FASTENER if fast else RHO_STRUCTURE
        link = table.get((leaf['sub'], leaf['name'])) or table.get((leaf['sub'], '*'))
        if leaf['sub'] == 'Wrist v3 LEFT' and leaf['name'] in ('Combine1', 'Cut-Revolve1[1]'):
            # pivot pins of the wrist pitch sit at z ~175; the third Combine1 is on the hand
            link = 'link7' if leaf['com'][2] < 150 else 'link5'
        if link is None:
            unassigned.append(leaf)
            if not fast and abs(leaf['vol']) > 100:
                print('note: %r in %r not in the part table; using nearest part' % (leaf['name'], leaf['path']),
                      file=sys.stderr)
        else:
            leaf['link'] = link
    anchors = [l for l in leaves if 'link' in l and l['bbox'] is not None]
    for leaf in unassigned:
        best = min(anchors, key=lambda a: point_box_distance(leaf['com'], a['bbox']))
        leaf['link'] = best['link']

    # -- wrist parallelogram: pins, rod ends, pitch pivot
    wrist = [l for l in leaves if l['sub'] == 'Wrist v3 LEFT']
    pins = sorted([l['com'] for l in wrist if l['name'] == 'Cut-Revolve1[1]'], key=lambda c: c[0])
    rod_ends = [l['com'] for l in wrist if l['name'] == 'Part 2']
    rod_a = max(rod_ends, key=lambda c: c[2])      # on the crank
    rod_b = min(rod_ends, key=lambda c: c[2])      # on the pitch lever
    pin_dir = (pins[1] - pins[0]) / np.linalg.norm(pins[1] - pins[0])
    a6 = joints['J6']['axis']
    pivot, skew = closest_between_lines(pins[0], a6, joints['J7']['center'], joints['J7']['axis'])
    c6 = joints['J6']['center']
    crank_r = np.linalg.norm((rod_a - c6) - a6 * ((rod_a - c6) @ a6))
    lever_r = np.linalg.norm((rod_b - pivot) - a6 * ((rod_b - pivot) @ a6))
    rod_len = np.linalg.norm((rod_b - rod_a) - a6 * ((rod_b - rod_a) @ a6))
    ground = np.linalg.norm((pivot - c6) - a6 * ((pivot - c6) @ a6))

    # -- elbow: CAD flexion about the J4 axis
    a3, a4, a5 = joints['J3']['axis'], joints['J4']['axis'], joints['J5']['axis']
    theta = signed_angle(-a3, -a5, a4)            # upper arm (down) to forearm (down)
    R_el = rotation(a4, -theta)
    c4 = joints['J4']['center']

    def straighten(p):
        p = np.asarray(p, float)
        return c4 + (p - c4) @ R_el.T

    def to_base(p):
        return (np.asarray(p, float) - ORIGIN_CAD) @ R_CB.T * 1e-3

    # joint points (CAD, after straightening where downstream of J4)
    pts = {
        'shoulder_pitch': joints['J1']['center'],
        'shoulder_roll': joints['J2']['center'],
        'shoulder_yaw': joints['J3']['center'],
        'elbow': c4,
        'forearm_yaw': straighten(joints['J5']['center']),
        'wrist_crank_joint': straighten(c6),
        'wrist_rod_joint': straighten(rod_a),
        'wrist_pitch': straighten(pivot),
        'wrist_yaw': straighten(pivot),
    }
    axes = {
        'shoulder_pitch': joints['J1']['axis'], 'shoulder_roll': joints['J2']['axis'],
        'shoulder_yaw': a3, 'elbow': a4, 'forearm_yaw': R_el @ a5,
        'wrist_crank_joint': R_el @ a6, 'wrist_rod_joint': R_el @ a6,
        'wrist_pitch': R_el @ a6, 'wrist_yaw': R_el @ joints['J7']['axis'],
    }
    tree = [('shoulder_pitch', 'base_link', 'link1'), ('shoulder_roll', 'link1', 'link2'),
            ('shoulder_yaw', 'link2', 'link3'), ('elbow', 'link3', 'link4'),
            ('forearm_yaw', 'link4', 'link5'), ('wrist_crank_joint', 'link5', 'wrist_crank'),
            ('wrist_rod_joint', 'wrist_crank', 'wrist_rod'), ('wrist_pitch', 'link5', 'link6'),
            ('wrist_yaw', 'link6', 'link7')]
    link_origin = {'base_link': np.zeros(3)}
    for jn, parent, child in tree:
        link_origin[child] = to_base(pts[jn])

    # -- tool0 at the peg tip, flange on the J7 output face
    peg = [l for l in leaves if l['name'] == 'J7PEG'][0]
    tri = triangles(peg['shape'], args.tol).reshape(-1, 3)
    tri = straighten(tri)
    centre = tri.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((tri - centre).T))
    peg_axis = v[:, -1]
    peg_axis = peg_axis if (centre - pts['wrist_yaw']) @ peg_axis > 0 else -peg_axis   # pointing away
    along = (tri - centre) @ peg_axis
    tip_pts = tri[along > along.max() - 0.3]
    tip = tip_pts.mean(axis=0)
    out_face = [m for m in inst['J7'] if abs(abs(m['vol']) - 6861) < 2.0][0]['com']
    flange = closest_point_on_line(pts['wrist_yaw'], axes['wrist_yaw'], straighten(out_face))

    # -- meshes and inertials per link
    os.makedirs(args.meshes, exist_ok=True)
    links = {}
    for link in ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'wrist_crank', 'wrist_rod',
                 'link6', 'link7']:
        members = [l for l in leaves if l.get('link') == link]
        R = R_CB @ (R_el if link in DOWNSTREAM_OF_ELBOW else np.eye(3))
        off = link_origin[link]

        def to_link(p_cad, link=link):
            p = straighten(p_cad) if link in DOWNSTREAM_OF_ELBOW else np.asarray(p_cad)
            return to_base(p) - off

        mass, first = 0.0, np.zeros(3)
        parts = []
        for m in members:
            vol = abs(m['vol'])
            mk = vol * m['rho']
            ck = to_link(m['com'])
            Ik = R @ (np.sign(m['vol']) * m['inertia'] * m['rho']) @ R.T * 1e-6   # kg m^2 about its COM
            parts.append((mk, ck, Ik))
            mass += mk
            first += mk * ck
        com = first / mass
        I = np.zeros((3, 3))
        for mk, ck, Ik in parts:
            d = ck - com
            I += Ik + mk * (d @ d * np.eye(3) - np.outer(d, d))
        tris = []
        for m in members:
            t = triangles(m['shape'], args.tol)
            if len(t):
                flat = t.reshape(-1, 3)
                tris.append(to_link(flat).reshape(-1, 3, 3))
        tris = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        write_stl(os.path.join(args.meshes, link + '.stl'), tris)
        names = sorted({'%s/%s' % (m['sub'], m['name']) for m in members
                        if not any(w in m['name'].lower() for w in FASTENER_WORDS)})
        links[link] = dict(mass_kg=mass, com_m=com.tolist(),
                           inertia=dict(ixx=I[0, 0], ixy=I[0, 1], ixz=I[0, 2], iyy=I[1, 1], iyz=I[1, 2],
                                        izz=I[2, 2]),
                           principal=np.linalg.eigvalsh(I).tolist(), solids=len(members),
                           triangles=int(len(tris)), parts=names)
        print('%-12s %6.3f kg  %4d solids  %7d triangles' % (link, mass, len(members), len(tris)),
              file=sys.stderr)

    result = dict(
        source=os.path.basename(args.step),
        base_origin_cad_mm=ORIGIN_CAD.tolist(),
        elbow_cad_flexion_deg=math.degrees(theta),
        joints={jn: dict(parent=p, child=c, origin_xyz=(link_origin[c] - link_origin[p]).tolist(),
                         axis=(R_CB @ axes[jn]).tolist(), point_base=link_origin[c].tolist())
                for jn, p, c in tree},
        actuators={j: dict(model=ACTUATOR_MODELS[SEEDS[j][0]], center_cad_mm=d['center'].tolist(),
                           axis_cad=d['axis'].tolist(), eig_ratio=d['eig_ratio'], solids=d['solids'])
                   for j, d in joints.items()},
        wrist_parallelogram=dict(crank_mm=crank_r, lever_mm=lever_r, rod_mm=rod_len, ground_mm=ground,
                                 pin_axis_vs_j6_deg=math.degrees(math.acos(min(1, abs(pin_dir @ a6)))),
                                 pivot_to_j7_axis_mm=skew),
        frames=dict(
            flange=dict(parent='link7', xyz=(to_base(flange) - link_origin['link7']).tolist(),
                        z_axis=(R_CB @ axes['wrist_yaw']).tolist(),
                        rpy=frame_rpy(R_CB @ axes['wrist_yaw'], (0, 0, -1))),
            tool0=dict(parent='link7', xyz=(to_base(tip) - link_origin['link7']).tolist(),
                       z_axis=(R_CB @ peg_axis).tolist(), rpy=frame_rpy(R_CB @ peg_axis, (1, 0, 0)),
                       peg_length_mm=float(along.max() - along.min()))),
        links=links,
        solids=[dict(name=l['name'], path=l['path'], link=l['link'], joint=l.get('joint'),
                     vol_mm3=round(abs(l['vol']), 1), com_cad_mm=[round(c, 2) for c in l['com']])
                for l in leaves],
    )
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ('elbow_cad_flexion_deg', 'wrist_parallelogram', 'frames')},
                     indent=1))
    for jn, d in result['joints'].items():
        print('%-18s %-11s -> %-11s xyz %s axis %s' % (jn, d['parent'], d['child'],
              np.round(d['origin_xyz'], 5).tolist(), np.round(d['axis'], 5).tolist()))


if __name__ == '__main__':
    main()
