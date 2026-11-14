"""Forward kinematics of arm01: URDF/Pinocchio against the CAD extraction, and
self-consistency of the FK, Jacobian and kinematic overlay.

The CAD numbers come from tools/cad/arm01_extract.json (written by
tools/cad/extract_arm01.py from the assembly STEP). Its `joints` block is
already in the URDF zero pose: everything past J4 rotated back by the CAD's
elbow flexion so q4 = 0 is a straight elbow. Its `actuators` and `solids`
blocks are raw CAD (mm, CAD axes, elbow flexed), so the FK at q4 = flexion
has to land on them without the straightening step.
"""

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from arm_stack.config import load_arm
from arm_stack.model.robot import ArmModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.path.join(ROOT, 'urdf', 'arm01', 'arm01.urdf')
with open(os.path.join(ROOT, 'tools', 'cad', 'arm01_extract.json')) as _f:
    CAD = json.load(_f)

POS_TOL = 1e-4                    # m: 0.1 mm
ANG_TOL = math.radians(0.05)      # rad
# CAD -> base_link (the URDF ledger: x forward = CAD -y, y left = CAD +x, z up),
# as in tools/cad/extract_arm01.py
R_CB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
ORIGIN_CAD_MM = np.array(CAD['base_origin_cad_mm'], float)
FLEXION = math.radians(CAD['elbow_cad_flexion_deg'])
ACTUATOR_JOINT = {'J1': 'shoulder_pitch', 'J2': 'shoulder_roll', 'J3': 'shoulder_yaw', 'J4': 'elbow',
                  'J5': 'forearm_yaw', 'J6': 'wrist_crank_joint', 'J7': 'wrist_yaw'}
LINK_FRAMES = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'flange', 'tool0']


def cad_to_base(p_mm):
    return R_CB @ (np.asarray(p_mm, float) - ORIGIN_CAD_MM) * 1e-3


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle(u, v):
    return math.acos(min(1.0, abs(float(unit(u) @ unit(v)))))


def signed_angle(u, v):
    return math.acos(max(-1.0, min(1.0, float(unit(u) @ unit(v)))))


def line_distance(p, point, direction):
    d = unit(direction)
    off = np.asarray(p, float) - point
    return float(np.linalg.norm(off - d * (off @ d)))


def solid(name, path_end='', joint=None, vol=None):
    out = [s for s in CAD['solids'] if s['name'] == name and s['path'].endswith(path_end)
           and (joint is None or s['joint'] == joint) and (vol is None or abs(s['vol_mm3'] - vol) < 2.0)]
    assert out, name
    return out


# --------------------------------------------------------------------------- models
@pytest.fixture(scope='module')
def full():
    """The whole URDF tree: the parallelogram's crank and rod are free joints here."""
    return pin.buildModelFromUrdf(URDF)


@pytest.fixture(scope='module')
def arm():
    return load_arm('config/arms/arm01.yaml')


@pytest.fixture()
def model(arm):
    return ArmModel(arm.urdf, arm.names)


def full_q(m, **values):
    q = pin.neutral(m)
    for name, v in values.items():
        q[m.joints[m.getJointId(name)].idx_q] = v
    return q


def _mimics(path):
    out = {}
    for j in ET.parse(path).getroot().findall('joint'):
        m = j.find('mimic')
        if m is not None:
            out[j.get('name')] = (m.get('joint'), float(m.get('multiplier', 1)), float(m.get('offset', 0)))
    return out


MIMIC = _mimics(URDF)


def wrist(q6, **values):
    """Full-model joint values with the URDF's mimic joints following wrist_pitch."""
    out = dict(values, wrist_pitch=q6)
    for name, (leader, mult, offset) in MIMIC.items():
        assert leader == 'wrist_pitch', name
        out[name] = mult * q6 + offset
    return out


def test_parallelogram_mimics_are_the_crank_and_rod():
    assert {n: m[1:] for n, m in MIMIC.items()} == {'wrist_crank_joint': (1.0, 0.0),
                                                    'wrist_rod_joint': (-1.0, 0.0)}


def joint_axis_world(d, j):
    S = np.array(d.joints[j].S).reshape(6, -1)
    return d.oMi[j].rotation @ S[3:, 0]


def kinematics(m, q):
    d = m.createData()
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    return d


def random_poses(model, count, seed):
    rng = np.random.default_rng(seed)
    return [rng.uniform(model.lower, model.upper) for _ in range(count)]


# =========================================================================== 1. FK vs CAD
@pytest.mark.parametrize('name', list(CAD['joints']))
def test_joint_axis_matches_cad_at_zero(full, name):
    """Every joint's axis point and direction (URDF zero, straight elbow)."""
    d = kinematics(full, pin.neutral(full))
    j = full.getJointId(name)
    cad = CAD['joints'][name]
    assert np.linalg.norm(d.oMi[j].translation - cad['point_base']) < POS_TOL
    a = joint_axis_world(d, j)
    assert float(unit(a) @ unit(cad['axis'])) > 0, 'axis sign flipped'
    assert angle(a, cad['axis']) < ANG_TOL


def test_driven_model_matches_cad_at_zero(model):
    """The reduced model (mimic joints locked) the sim and controller use."""
    d = kinematics(model.model, model._q(np.zeros(model.n)))
    for k, (name, j) in enumerate(zip(model.joint_names, model.jid)):
        cad = CAD['joints'][name]
        assert np.linalg.norm(d.oMi[j].translation - cad['point_base']) < POS_TOL, name
        a = d.oMi[j].rotation @ model.axis_in_child(k)
        assert float(a @ unit(cad['axis'])) > 0 and angle(a, cad['axis']) < ANG_TOL, name


def test_elbow_zero_is_straight_and_flexion_is_the_cad_pose(full):
    j3, j4, j5 = (full.getJointId(n) for n in ('shoulder_yaw', 'elbow', 'forearm_yaw'))
    d = kinematics(full, pin.neutral(full))
    assert angle(joint_axis_world(d, j5), joint_axis_world(d, j3)) < ANG_TOL
    d = kinematics(full, full_q(full, elbow=FLEXION))
    a3, a4, a5 = (joint_axis_world(d, j) for j in (j3, j4, j5))
    assert abs(signed_angle(a3, a5) - FLEXION) < ANG_TOL
    # positive q4 flexes the forearm forward (+x), per the ledger
    forearm = d.oMi[j5].translation - d.oMi[j4].translation
    d0 = kinematics(full, pin.neutral(full))
    assert forearm[0] > (d0.oMi[j5].translation - d0.oMi[j4].translation)[0]


@pytest.mark.parametrize('act', list(ACTUATOR_JOINT))
def test_cad_pose_lands_on_raw_actuator_axes(full, act):
    """At q4 = CAD flexion the FK reproduces the un-straightened CAD: each
    actuator's cylinder axis (centre and direction) from the STEP."""
    d = kinematics(full, full_q(full, elbow=FLEXION))
    j = full.getJointId(ACTUATOR_JOINT[act])
    raw = CAD['actuators'][act]
    centre = cad_to_base(raw['center_cad_mm'])
    a = joint_axis_world(d, j)
    assert angle(a, R_CB @ np.array(raw['axis_cad'])) < ANG_TOL
    if act == 'J7':
        # link7's origin is the wrist-pitch pivot, which the extractor puts on
        # the common perpendicular of the (0.2 mm skew) pitch-pin line and the
        # J7 axis, so the URDF's J7 axis sits half the skew off the CAD's
        skew = CAD['wrist_parallelogram']['pivot_to_j7_axis_mm'] * 1e-3
        assert line_distance(centre, d.oMi[j].translation, a) < 0.5 * skew + 0.02e-3
    else:
        assert np.linalg.norm(d.oMi[j].translation - centre) < POS_TOL


# --------------------------------------------------------------------------- wrist parallelogram
def parallelogram_points(full):
    """Crank pivot C, crank pin A, pitch pivot P, lever pin B in base_link at
    the CAD pose (q4 = flexion), plus the pin axis."""
    d = kinematics(full, full_q(full, elbow=FLEXION))
    jc, jr, j6 = (full.getJointId(n) for n in ('wrist_crank_joint', 'wrist_rod_joint', 'wrist_pitch'))
    ends = [s['com_cad_mm'] for s in solid('Part 2', 'Wrist v3 LEFT')]
    B = cad_to_base(min(ends, key=lambda c: c[2]))          # rod end on the pitch lever
    A_cad = cad_to_base(max(ends, key=lambda c: c[2]))      # rod end on the crank
    return d, d.oMi[jc].translation, d.oMi[jr].translation, d.oMi[j6].translation, B, A_cad, \
        unit(joint_axis_world(d, j6))


def test_parallelogram_geometry_matches_cad(full):
    d, C, A, P, B, A_cad, a = parallelogram_points(full)
    perp = lambda v: np.linalg.norm(v - a * (v @ a))   # noqa: E731
    ref = CAD['wrist_parallelogram']
    assert np.linalg.norm(A - A_cad) < POS_TOL              # URDF rod joint on the CAD rod end
    assert abs(perp(A - C) * 1e3 - ref['crank_mm']) < 0.1
    assert abs(perp(P - C) * 1e3 - ref['ground_mm']) < 0.1
    assert abs(perp(B - A) * 1e3 - ref['rod_mm']) < 0.1
    assert abs(perp(B - P) * 1e3 - ref['lever_mm']) < 0.1
    # all three pin axes parallel to the wrist pitch
    for n in ('wrist_crank_joint', 'wrist_rod_joint'):
        assert angle(joint_axis_world(d, full.getJointId(n)), a) < ANG_TOL


def test_parallelogram_mimic_keeps_closure_over_pitch_range(full):
    """Crank = +q6 and rod = -q6 (the URDF mimics): the rod keeps link5's
    orientation and its far end stays on link6's lever pin. The CAD linkage is
    not an exact parallelogram (rod 70.00 vs ground 69.89 mm), so the pin
    drifts apart by 2 sin(q6/2) times that mismatch, under 0.2 mm at 90 deg."""
    d0, C, A, P, B, _, a = parallelogram_points(full)
    j5, jr, j6 = (full.getJointId(n) for n in ('forearm_yaw', 'wrist_rod_joint', 'wrist_pitch'))
    B_rod = d0.oMi[jr].actInv(B)           # lever pin, as a point on the rod
    B_l6 = d0.oMi[j6].actInv(B)            # ... and as a point on link6
    worst = 0.0
    for deg in np.linspace(-90, 90, 37):
        q6 = math.radians(deg)
        d = kinematics(full, full_q(full, **wrist(q6, elbow=FLEXION)))
        rel = d.oMi[j5].rotation.T @ d.oMi[jr].rotation
        assert np.linalg.norm(pin.log3(rel)) < 1e-12, 'rod turned with the crank'
        crank_pin = d.oMi[jr].translation
        assert abs(np.linalg.norm(crank_pin - C) - np.linalg.norm(A - C)) < 1e-12
        gap = np.linalg.norm(d.oMi[jr].act(B_rod) - d.oMi[j6].act(B_l6))
        worst = max(worst, gap)
        assert gap < 0.2e-3, 'parallelogram opens by %.3f mm at q6 = %.0f deg' % (gap * 1e3, deg)
    assert worst > 0.05e-3   # the CAD mismatch is really there; the check above is not vacuous


def test_parallelogram_is_one_to_one_within_joint_limits(full):
    """Solve the real four-bar (CAD pin positions) for the lever angle at each
    crank angle: inside the wrist_pitch limits the 1:1 mimic is within 0.25 deg."""
    _, C, A, P, B, _, a = parallelogram_points(full)
    perp = lambda v: v - a * (v @ a)   # noqa: E731
    c0, l0, L = perp(A - C), perp(B - P), np.linalg.norm(perp(B - A))
    j6 = full.getJointId('wrist_pitch')
    lo, hi = full.lowerPositionLimit[full.joints[j6].idx_q], full.upperPositionLimit[full.joints[j6].idx_q]

    def rot(t):
        return pin.exp3(a * t)

    for phi in np.linspace(lo, hi, 29):
        d = perp(C + rot(phi) @ c0 - P)
        # |R(psi) l0 - d| = L  ->  cos(psi) l0.d + sin(psi) (a x l0).d = k
        k = (l0 @ l0 + d @ d - L * L) / 2
        x, y = l0 @ d, np.cross(a, l0) @ d
        cands = [math.atan2(y, x) + s * math.acos(k / math.hypot(x, y)) for s in (1, -1)]
        psi = min(cands, key=lambda c: abs(math.remainder(c - phi, 2 * math.pi)))
        err = math.remainder(psi - phi, 2 * math.pi)
        assert abs(err) < math.radians(0.25), 'lever %.3f deg off the crank at %.1f deg' % (
            math.degrees(err), math.degrees(phi))


# --------------------------------------------------------------------------- flange and tool0
def test_flange_and_tool0_match_cad_frames(model):
    d = kinematics(model.model, model._q(np.zeros(model.n)))
    l7 = np.array(CAD['joints']['wrist_yaw']['point_base'])
    j7 = model.jid[model.joint_names.index('wrist_yaw')]
    a7 = d.oMi[j7].rotation @ model.axis_in_child(model.joint_names.index('wrist_yaw'))
    for name, x_hint in (('flange', (0, 0, -1)), ('tool0', (1, 0, 0))):
        M = d.oMf[model.model.getFrameId(name)]
        ref = CAD['frames'][name]
        assert np.linalg.norm(M.translation - (l7 + ref['xyz'])) < POS_TOL, name
        assert float(M.rotation[:, 2] @ unit(ref['z_axis'])) > math.cos(ANG_TOL), name
        assert float(M.rotation[:, 0] @ np.array(x_hint)) > 0, name     # ledger x convention
    flange = d.oMf[model.model.getFrameId('flange')]
    # the flange sits on the J7 axis, z out along it
    assert line_distance(flange.translation, d.oMi[j7].translation, a7) < POS_TOL
    assert float(flange.rotation[:, 2] @ unit(a7)) > math.cos(ANG_TOL)


def test_flange_on_the_raw_j7_output_face(full):
    """At the CAD pose, the flange origin is in the plane of J7's output-face
    solid (raw STEP centroid, independent of the extractor's frame math)."""
    d = kinematics(full, full_q(full, elbow=FLEXION))
    M = d.oMf[full.getFrameId('flange')]
    face = cad_to_base(solid('X4-P12.5-10', 'Wrist v3 LEFT', joint='J7', vol=6861.0)[0]['com_cad_mm'])
    assert abs(float((face - M.translation) @ M.rotation[:, 2])) < POS_TOL


def test_tool0_is_rigid_on_the_flange(model):
    ref = None
    for q in random_poses(model, 20, seed=3):
        T = model.fk(q, 'flange').actInv(model.fk(q, 'tool0'))
        if ref is None:
            ref = T
        assert np.allclose(T.homogeneous, ref.homogeneous, atol=1e-12)


# =========================================================================== 2. self-consistency
def overlay_model(arm, overlay):
    m = ArmModel(arm.urdf, arm.names)
    m.apply_overlay(overlay)
    return m


@pytest.mark.parametrize('frame', ['tool0', 'flange', 'link4'])
def test_jacobian_matches_finite_differences(model, arm, frame):
    calibrated = overlay_model(arm, {
        'shoulder_roll': {'dxyz': [0.001, -0.002, 0.0005], 'drot': [0.01, 0, -0.02]},
        'wrist_pitch': {'dxyz': [0.0, 0.001, 0.0], 'drot': [0.0, 0.015, 0.0]}})
    h = 1e-6
    for m in (model, calibrated):
        for q in random_poses(m, 10, seed=1):
            J = m.jacobian(q, frame)
            fd = np.zeros_like(J)
            for k in range(m.n):
                dq = np.zeros(m.n)
                dq[k] = h
                Mp, Mm = m.fk(q + dq, frame), m.fk(q - dq, frame)
                fd[:3, k] = (Mp.translation - Mm.translation) / (2 * h)
                fd[3:, k] = pin.log3(Mp.rotation @ Mm.rotation.T) / (2 * h)
            assert np.allclose(J, fd, atol=1e-7), np.abs(J - fd).max()


def subtree(m, root):
    out = {root}
    for j in range(root + 1, m.njoints):
        if m.parents[j] in out:
            out.add(j)
    return out


@pytest.mark.parametrize('name', ['shoulder_pitch', 'shoulder_yaw', 'elbow', 'wrist_pitch', 'wrist_yaw'])
def test_overlay_translation_moves_downstream_frames_by_that_amount(model, arm, name):
    dxyz = np.array([0.0012, -0.0007, 0.0021])
    shifted = overlay_model(arm, {name: {'dxyz': dxyz.tolist()}})
    k = model.joint_names.index(name)
    j = model.jid[k]
    down = subtree(model.model, j)
    frames = [f for f in model.model.frames]
    for q in [np.zeros(model.n)] + random_poses(model, 10, seed=2):
        a = kinematics(model.model, model._q(q))
        b = kinematics(shifted.model, shifted._q(q))
        # the shift is along dxyz in the parent joint's frame (in base_link at q = 0)
        R_parent = a.oMi[model.model.parents[j]].rotation
        expect = R_parent @ dxyz
        if not q.any():
            assert np.allclose(expect, dxyz, atol=1e-12)
        for fid, f in enumerate(frames):
            Ma, Mb = a.oMf[fid], b.oMf[fid]
            assert np.allclose(Ma.rotation, Mb.rotation, atol=1e-12), f.name
            moved = Mb.translation - Ma.translation
            want = expect if f.parentJoint in down else np.zeros(3)
            assert np.allclose(moved, want, atol=1e-12), f.name


def test_overlay_rotation_pivots_on_the_joint_origin(model, arm):
    name = 'elbow'
    tilted = overlay_model(arm, {name: {'drot': [0.02, -0.01, 0.015]}})
    j = model.jid[model.joint_names.index(name)]
    down = subtree(model.model, j)
    for q in random_poses(model, 10, seed=4):
        a = kinematics(model.model, model._q(q))
        b = kinematics(tilted.model, tilted._q(q))
        o = a.oMi[j].translation
        assert np.allclose(b.oMi[j].translation, o, atol=1e-12)
        for fid, f in enumerate(model.model.frames):
            pa, pb = a.oMf[fid].translation, b.oMf[fid].translation
            if f.parentJoint in down:
                assert abs(np.linalg.norm(pb - o) - np.linalg.norm(pa - o)) < 1e-12, f.name
            else:
                assert np.allclose(pa, pb, atol=1e-12), f.name


# --------------------------------------------------------------------------- independent FK
def _floats(text, default):
    return np.array([float(v) for v in (text or default).split()])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                    # URDF: fixed-axis roll, pitch, yaw


def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp_twist(w, v, theta):
    """exp([S] theta) for a unit-rotation screw S = (w, v)."""
    K = _hat(w)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
    T[:3, 3] = (np.eye(3) * theta + (1 - math.cos(theta)) * K + (theta - math.sin(theta)) * K @ K) @ v
    return T


class ProductOfExponentials:
    """Space-frame PoE FK built straight from the URDF XML (no Pinocchio):
    T(q) = exp([S1] q1) ... exp([Sn] qn) M, with M the zero-pose link transform
    and S the joint screws in base_link at the zero pose."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joint = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            ax = j.find('axis')
            mimic = j.find('mimic')
            T = np.eye(4)
            T[:3, :3] = _rpy(*_floats(o.get('rpy') if o is not None else None, '0 0 0'))
            T[:3, 3] = _floats(o.get('xyz') if o is not None else None, '0 0 0')
            self.joint[j.get('name')] = dict(
                type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                origin=T, axis=unit(_floats(ax.get('xyz') if ax is not None else None, '1 0 0')),
                mimic=None if mimic is None else (mimic.get('joint'), float(mimic.get('multiplier', 1)),
                                                  float(mimic.get('offset', 0))))
        self.by_child = {d['child']: n for n, d in self.joint.items()}
        self.M = {}
        self.screw = {}
        for link in [d['child'] for d in self.joint.values()]:
            self._zero(link)
        for n, d in self.joint.items():
            if d['type'] in ('revolute', 'continuous'):
                Mc = self.M[d['child']]
                w = Mc[:3, :3] @ d['axis']
                self.screw[n] = (w, -np.cross(w, Mc[:3, 3]))

    def _zero(self, link):
        if link not in self.M:
            if link not in self.by_child:
                self.M[link] = np.eye(4)            # root
            else:
                d = self.joint[self.by_child[link]]
                self.M[link] = self._zero(d['parent']) @ d['origin']
        return self.M[link]

    def chain(self, link):
        out = []
        while link in self.by_child:
            n = self.by_child[link]
            out.append(n)
            link = self.joint[n]['parent']
        return out[::-1]

    def fk(self, q: dict, link):
        T = np.eye(4)
        for n in self.chain(link):
            if n not in self.screw:
                continue
            m = self.joint[n]['mimic']
            theta = m[1] * q[m[0]] + m[2] if m else q[n]
            T = T @ _exp_twist(*self.screw[n], theta)
        return T @ self.M[link]


def test_independent_poe_fk_matches_pinocchio(model, full):
    poe = ProductOfExponentials(URDF)
    for q in random_poses(model, 100, seed=5):
        qmap = dict(zip(model.joint_names, q))
        for frame in LINK_FRAMES:
            err = np.abs(model.fk(q, frame).homogeneous - poe.fk(qmap, frame)).max()
            assert err < 1e-9, (frame, err)
        # the parallelogram's moving parts, with the mimic joints following
        d = kinematics(full, full_q(full, **wrist(qmap['wrist_pitch'],
                                                   **{n: v for n, v in qmap.items() if n != 'wrist_pitch'})))
        for frame in ('wrist_crank', 'wrist_rod'):
            err = np.abs(d.oMf[full.getFrameId(frame)].homogeneous - poe.fk(qmap, frame)).max()
            assert err < 1e-9, (frame, err)


def test_overlay_applied_twice_does_not_compound():
    arm = load_arm()
    m = ArmModel(arm.urdf, arm.names)
    overlay = {arm.names[3]: {'dxyz': [0.001, -0.0005, 0.0007], 'drot': [0.002, 0.0, -0.001]}}
    q = np.linspace(-0.3, 0.4, m.n)
    m.apply_overlay(overlay)
    once = m.fk(q).homogeneous.copy()
    m.apply_overlay(overlay)
    np.testing.assert_allclose(m.fk(q).homogeneous, once, atol=1e-12)
    # and the nominal model is untouched
    plain = ArmModel(arm.urdf, arm.names)
    np.testing.assert_allclose(plain.fk(q).homogeneous,
                               ArmModel(arm.urdf, arm.names).fk(q).homogeneous, atol=1e-12)
    assert not np.allclose(plain.fk(q).homogeneous, once, atol=1e-6)
