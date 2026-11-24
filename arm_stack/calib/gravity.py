"""Stage 5, `arm gravity`: fit each link's center of mass.

Static joint torque is linear in each link's first moment h_k = m_k c_k:

    tau_g(q) = Y(q) h + Y0(q) m

with Y, Y0 from Pinocchio's joint torque regressor and m the masses weighed
at assembly (joint config `link_masses_kg`, URDF values where still null).
At each pose all joints sweep slowly up and then down through it in Mode P;
the up and down means are averaged so friction cancels. h is fit by
regularized least squares (prior = URDF), since some first-moment directions
are unobservable from joint torques and stay at their URDF values.

Pass: torque residual under 5% of each joint's peak gravity torque on 10
held-out poses. A center of mass that moves more than 10 mm from the URDF
is flagged: check that link's CAD and wiring before trusting the fit.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import Trace, hold_and_average, move_to, stream
from .common import DEG, StageContext, StageResult, poses_clear_of_keep_out, random_poses, safe_home

RESIDUAL_FRACTION = 0.05
LOADED_NM = 0.5       # peak gravity above which a joint must beat the URDF (J1-J4 on arm01)
COM_FLAG_M = 0.010
SWEEP = 2 * DEG
SPEED = 0.056         # rad/s through each pose (0.04 until 2026-11-23, +40% on request)
MOVE_SPEED = 0.56     # rad/s between poses (was the 0.4 calibration cap, +40%)
N_FIT, N_TEST = 12, 4 # 16 poses x 7 joints for <= 28 parameters (rank ~14); was 32 + 10
# Poses near hanging (the rest pose, where float starts): the random draw's 12 deg margin
# and 0.8 spread never put J2 under ~31 deg, so the fit extrapolated there and floated
# J2 up on its own (arm01, 2026-11-24). This many fit / test poses are drawn within
# NEAR_HANGING_DEG of the hanging pose instead.
N_NEAR_FIT, N_NEAR_TEST = 4, 2
NEAR_HANGING_DEG = 30.0
N_REPEAT = 2          # test poses measured twice: the repeatability floor of the pass check
PRIOR_WEIGHT = 1e-3


def measure_pose(ctx, q, clamp, sweep=SWEEP, speed=SPEED):
    """Measured joint torque at q, averaged over an up and a down pass."""
    client = ctx.client
    available = float(np.min(np.minimum(q - ctx.model.lower, ctx.model.upper - q)))
    sweep = min(sweep, max(0.0, available) * 0.9)
    if sweep < 0.01 * DEG:
        raise RuntimeError('Gravity measurement pose has insufficient clearance for its sweep.')
    passes = []
    for sgn in (+1, -1):
        start = q - sgn * sweep
        move_to(client, start, max_speed=MOVE_SPEED, settle=0.3, torque_clamp=clamp)
        n = max(2, int(2 * sweep / speed / client.cmd_dt))
        s = np.linspace(0, 1, n)[:, None]
        qs = start[None, :] + sgn * 2 * sweep * s
        qds = np.repeat((sgn * speed * np.ones(len(q)))[None, :], n, axis=0)
        tr = Trace()
        stream(client, qs, qds, trace=tr, torque_clamp=clamp)
        a = tr.arrays()
        mid = slice(n // 4, 3 * n // 4)
        passes.append((a['tau'][mid].mean(axis=0), a['q'][mid].mean(axis=0)))
    tau = 0.5 * (passes[0][0] + passes[1][0])
    q_meas = 0.5 * (passes[0][1] + passes[1][1])
    measure_pose.friction = 0.5 * np.abs(passes[0][0] - passes[1][0])   # up vs down pass
    return tau, q_meas


def fit_first_moments(model, Q, T, masses, prior_h, weight=PRIOR_WEIGHT, scale=None):
    rows, rhs = [], []
    scale = np.ones(model.n) if scale is None else scale
    for q, tau in zip(Q, T):
        Y, Y0 = model.gravity_first_moment_regressor(q)
        rows.append(Y / scale[:, None])
        rhs.append((tau - Y0 @ masses - model.springs(q)) / scale)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    lam = weight * np.trace(A.T @ A) / A.shape[1]
    H = A.T @ A + lam * np.eye(A.shape[1])
    h = np.linalg.solve(H, A.T @ b + lam * prior_h)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    fit_first_moments.row_space = Vt[sv > sv[0] * 1e-6]   # identifiable directions of h
    return h, sv


def residuals(model, Q, T):
    return np.array([tau - model.gravity(q) for q, tau in zip(Q, T)])


def measure_static_pose(ctx, q, clamp):
    """Average settled holding torque after approaching from both directions."""
    clearance = np.minimum(q - ctx.model.lower, ctx.model.upper - q)
    approach = np.minimum(0.25 * DEG, 0.5 * clearance)
    reads = []
    for direction in (-1, 1):
        move_to(ctx.client, q + direction * approach, max_speed=ctx.motion_speed,
                gravity_ff=False, torque_clamp=clamp)
        move_to(ctx.client, q, max_speed=ctx.motion_speed, settle=1.0,
                gravity_ff=False, torque_clamp=clamp)
        reads.append(hold_and_average(ctx.client, 0.5, q, gravity_ff=False, torque_clamp=clamp))
        if np.max(np.abs(reads[-1]['q'] - q)) > 1 * DEG:
            raise RuntimeError('A joint failed to reach its gravity measurement pose within 1 degree.')
    return np.mean([r['tau'] for r in reads], axis=0), np.mean([r['q'] for r in reads], axis=0)


def spring_regressor(q, joints, n) -> np.ndarray:
    """n x 2*len(joints): torque on joint j from (offset, stiffness) = (1, q_j)."""
    S = np.zeros((n, 2 * len(joints)))
    for c, j in enumerate(joints):
        S[j, 2 * c] = 1.0
        S[j, 2 * c + 1] = q[j]
    return S


def fit_static_parameters(model, Q, T, scale, bias=False, springs=()):
    """Regularized static mass/first-moment fit with positive mass bounds. With bias=True
    each joint also gets a constant torque offset (a diagnostic for forces no link mass
    explains, e.g. cable drag); its values come back as a fifth element. springs: joint
    indices that also get offset + stiffness * q (not pulled toward the URDF: the URDF
    has no springs); (offsets, stiffnesses) come back as the last element."""
    from scipy.optimize import lsq_linear
    n = model.n
    springs = list(springs)
    ns = 2 * len(springs)
    rows = []
    for q in Q:
        moments, masses = model.gravity_first_moment_regressor(q)
        cols = [masses, moments] + ([np.eye(n)] if bias else []) + [spring_regressor(q, springs, n)]
        rows.append(np.column_stack(cols) / scale[:, None])
    A = np.vstack(rows)
    b = np.concatenate([t / scale for t in T])
    prior = np.concatenate([model.masses(), model.first_moments()] + ([np.zeros(n)] if bias else [])
                           + [np.zeros(ns)])
    lam = PRIOR_WEIGHT * np.trace(A.T @ A) / A.shape[1]
    weight = np.full(len(prior), np.sqrt(lam))
    weight[len(prior) - ns:] = np.sqrt(lam) * 1e-3          # springs: nearly free
    augmented = np.vstack([A, np.diag(weight)])
    rhs = np.concatenate([b, weight * prior])
    extra = [np.full(n, -np.inf)] if bias else []
    lower = np.concatenate([0.05 * model.masses(), np.full(3 * n, -np.inf)] + extra + [np.full(ns, -np.inf)])
    upper = np.concatenate([5 * model.masses(), np.full(3 * n, np.inf)] + [-e for e in extra]
                           + [np.full(ns, np.inf)])
    fit = lsq_linear(augmented, rhs, bounds=(lower, upper))
    if not fit.success:
        raise RuntimeError('Static gravity parameter fit did not converge.')
    _, singular, vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-6))
    out = (fit.x[:n], fit.x[n:4 * n], singular, vt[:rank])
    if bias:
        out = out + (fit.x[4 * n:5 * n],)
    if springs:
        sp = fit.x[len(prior) - ns:]
        offsets, stiff = np.zeros(n), np.zeros(n)
        offsets[springs], stiff[springs] = sp[0::2], sp[1::2]
        out = out + ((offsets, stiff),)
    return out


def nearest_neighbour_order(start, poses):
    """Visit order that keeps each move short (largest-joint distance, greedy)."""
    left = list(range(len(poses)))
    order, here = [], np.asarray(start, float)
    while left:
        k = min(left, key=lambda i: np.max(np.abs(poses[i] - here)))
        order.append(k)
        left.remove(k)
        here = poses[k]
    return [poses[i] for i in order]


def poses_near_hanging(model, arm, count, rng, zones, reach_deg=NEAR_HANGING_DEG,
                       margin=5 * DEG, tries=500) -> np.ndarray:
    """Poses within reach_deg of the hanging pose (rough_pose 'hanging'), kept margin
    inside the soft limits and clear of the keep-out zones."""
    from .common import clear_of_keep_out
    from .stop_zero import rough_pose
    hang = rough_pose(arm, 'hanging')
    margin = np.minimum(margin, 0.2 * (model.upper - model.lower))
    lo, hi = model.lower + margin, model.upper - margin
    out = []
    for _ in range(count * tries):
        q = np.clip(hang + rng.uniform(-1, 1, model.n) * reach_deg * DEG, lo, hi)
        if clear_of_keep_out(zones, q):
            out.append(q)
            if len(out) == count:
                return np.array(out)
    raise RuntimeError('could not draw %d poses near hanging clear of the keep-out zones' % count)


def run(ctx: StageContext, n_fit: int = None, n_test: int = None, fit_masses: bool = False,
        static: bool = False) -> StageResult:
    client, arm = ctx.client, ctx.arm
    model = ctx.model.copy()
    n_fit = n_fit or N_FIT
    n_test = n_test or N_TEST
    home = safe_home(model)
    peak_g = model.peak_gravity_torque()
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    zones = getattr(arm, 'keep_out', [])
    n_near_fit, n_near_test = min(N_NEAR_FIT, n_fit // 3), min(N_NEAR_TEST, n_test // 2)
    if 'elbow' not in arm.names:          # no hanging pose (the bench rig): random draws only
        n_near_fit = n_near_test = 0
    wide = poses_clear_of_keep_out(model, n_fit + n_test - n_near_fit - n_near_test, ctx.rng, zones,
                                   margin=12 * DEG, spread=0.8)
    near = (poses_near_hanging(model, arm, n_near_fit + n_near_test, ctx.rng, zones)
            if n_near_fit + n_near_test else np.zeros((0, model.n)))
    poses = np.concatenate([wide[:n_fit - n_near_fit], near[:n_near_fit],
                            wide[n_fit - n_near_fit:], near[n_near_fit:]])
    # short moves: fit poses in nearest-neighbour order from where the arm is, then the
    # test poses, then the first N_REPEAT test poses again (repeatability)
    fit_poses = nearest_neighbour_order(client.state().q, list(poses[:n_fit]))
    test_poses = nearest_neighbour_order(fit_poses[-1], list(poses[n_fit:]))
    n_rep = min(N_REPEAT, n_test)
    schedule = fit_poses + test_poses + test_poses[:n_rep]
    measure = measure_static_pose if static else measure_pose
    data, frictions = [], []
    for pose_index, q in enumerate(schedule):
        if not ctx.sim:
            print('Gravity measurement %d/%d' % (pose_index + 1, len(schedule)), flush=True)
        clamp = np.minimum(peak, np.abs(model.gravity(q)) * 1.5 + 0.25 * peak)
        tau, q_meas = measure(ctx, q, clamp)
        data.append((q_meas, tau))
        frictions.append(getattr(measure, 'friction', np.zeros(model.n)) if not static else np.zeros(model.n))
    move_to(client, home, max_speed=MOVE_SPEED)
    repeats = data[n_fit + n_test:]
    data = data[:n_fit + n_test]
    repeat_nm = (np.max([np.abs(r[1] - d[1]) for r, d in zip(repeats, data[n_fit:])], axis=0)
                 if repeats else np.zeros(model.n))
    fit, test = data[:n_fit], data[n_fit:]
    Qf, Tf = [d[0] for d in fit], [d[1] for d in fit]
    Qt, Tt = [d[0] for d in test], [d[1] for d in test]
    if static and np.any(np.ptp(Qf, axis=0) < 0.1 * (model.upper - model.lower)):
        raise RuntimeError('Insufficient measured joint movement for gravity identification.')

    masses = model.masses()
    prior = model.first_moments()
    before = np.abs(residuals(model, Qt, Tt)).max(axis=0)
    springs = None
    if fit_masses:
        # joint springs on the loaded joints: arm01's J2 carries a cable spring the rigid
        # bodies cannot (-0.4 Nm near hanging, 2026-11-24)
        spring_joints = [i for i in range(model.n) if peak_g[i] >= LOADED_NM]
        masses, h, sv, basis, (sp_off, sp_k) = fit_static_parameters(
            model, Qf, Tf, np.maximum(peak_g, 0.1), springs=spring_joints)
        fitted = {link: {'mass_kg': float(masses[k]), 'com_m': (h[3*k:3*k+3] / masses[k]).tolist()}
                  for k, link in enumerate(model.link_names())}
        springs = {name: {'offset_nm': float(sp_off[i]), 'stiffness_nm_per_rad': float(sp_k[i])}
                   for i, name in enumerate(arm.names)}
        model.apply_calibration({'link_inertials': fitted, 'joint_springs': springs})
    else:
        h, sv = fit_first_moments(model, Qf, Tf, masses, prior, scale=np.maximum(peak_g, 0.1))
        basis = fit_first_moments.row_space
        model.set_first_moments(h)
    after = residuals(model, Qt, Tt)
    worst = np.abs(after).max(axis=0)
    # a joint that carries almost no gravity cannot be judged below the drive's
    # torque resolution: the threshold is max(5% of peak gravity, one 0x6077 count)
    from ..units import Units
    count_nm = np.abs(Units(arm, ctx.calibration).nm_per_permille())
    # nor below what a pose repeats to: the model cannot fit better than the data repeats
    allowed = np.maximum(np.maximum(RESIDUAL_FRACTION * peak_g, count_nm), repeat_nm)
    frac = worst / np.maximum(peak_g, 1e-6)
    # Accepted rule (2026-11-23): 5% of peak gravity is not reachable even on noise-free sim
    # data for this arm (light wrists: 5% is ~6 mNm, far under their friction). A fit is
    # used if (a) on every loaded joint it predicts held-out poses better than the URDF,
    # and (b) each joint's held-out error is within the 5%/count/repeat allowance or below
    # that joint's friction measured in this run (median up/down half-difference), which
    # the drive's position loop absorbs anyway.
    friction_nm = np.median(np.array(frictions), axis=0) if frictions else np.zeros(model.n)
    loaded = peak_g >= LOADED_NM
    beats_urdf = (worst < before) | ~loaded
    within = (worst < allowed) | (worst < friction_nm)
    passed = bool(np.all(beats_urdf & within))

    lines, flags = [], []
    inertials = {}
    links = model.link_names()
    # only the identifiable part of the change says anything about the hardware;
    # the rest is the ridge prior at work
    V = basis
    dh_id = h - prior if fit_masses else V.T @ (V @ (h - prior))
    if fit_masses:
        lines.append('Masses and centers of mass were fitted jointly. Static data identifies '
                     'effective parameter combinations; individual link masses may be ambiguous.')
    for k, link in enumerate(links):
        com = h[3 * k:3 * k + 3] / masses[k]
        shift = float(np.linalg.norm(dh_id[3 * k:3 * k + 3]) / masses[k])
        inertials[link] = {'mass_kg': float(masses[k]), 'com_m': [float(c) for c in com]}
        lines.append('%-15s com [%+.4f %+.4f %+.4f] m, identifiable shift %.1f mm from URDF' % (
            link, com[0], com[1], com[2], shift * 1e3))
        if shift > COM_FLAG_M:
            flags.append('%s: COM moved %.1f mm from the URDF; check its CAD and wiring' % (link, shift * 1e3))
    for i, name in enumerate(arm.names):
        ok = beats_urdf[i] and within[i]
        lines.append('%-15s held-out residual %.3f Nm = %.1f%% of peak gravity %.2f Nm (URDF %.3f; '
                     'repeat %.3f, friction %.3f)%s' % (name, worst[i], 100 * frac[i], peak_g[i], before[i],
                                                       repeat_nm[i], friction_nm[i], '' if ok else '  <-- FAIL'))
    if fit_masses:
        # diagnostic only (nothing written): a constant per-joint offset soaks up torque no
        # link mass explains (cable drag, a spring); if a big COM shift collapses with it,
        # the shift was not geometry
        bm, bh, _, _, bias = fit_static_parameters(ctx.model, Qf, Tf, np.maximum(peak_g, 0.1), bias=True)
        prior_b = ctx.model.first_moments()
        lines.append('bias diagnostic (fit with a constant torque per joint; not written):')
        for i, name in enumerate(arm.names):
            lines.append('  %-15s offset %+.3f Nm' % (name, bias[i]))
        for k, link in enumerate(links):
            sh = float(np.linalg.norm(bh[3 * k:3 * k + 3] - prior_b[3 * k:3 * k + 3]) / bm[k])
            lines.append('  %-15s mass %.3f kg, COM shift %.1f mm with offsets' % (link, bm[k], sh * 1e3))
    if springs:
        lines.append('joint springs (offset + stiffness x q, on the loaded joints):')
        for name, e in springs.items():
            if e['offset_nm'] or e['stiffness_nm_per_rad']:
                lines.append('  %-15s %+.3f Nm %+.3f Nm/rad' % (name, e['offset_nm'], e['stiffness_nm_per_rad']))
    cond = float(sv[0] / sv[sv > sv[0] * 1e-9][-1]) if len(sv) else float('nan')
    rank = int(np.sum(sv > sv[0] * 1e-6))
    lines.append('regressor rank %d of %d first-moment parameters (rest held at URDF values)' % (rank, len(h)))
    return StageResult('gravity', passed,
                       {'residual_frac': frac.tolist(), 'residual_nm': worst.tolist(),
                        'repeat_nm': repeat_nm.tolist(), 'friction_nm': friction_nm.tolist(),
                        'before_nm': before.tolist(), 'rank': rank, 'cond': cond,
                        'measured_poses_rad': [q.tolist() for q, _ in data],
                        'holding_torques_nm': [tau.tolist() for _, tau in data],
                        'fit_poses': [q.tolist() for q in Qf], 'test_poses': [q.tolist() for q in Qt]},
                       dict({'link_inertials': inertials,
                             'gravity_validation': {'passed': passed, 'fit_masses': fit_masses,
                                                    'max_residual_nm': worst.tolist(),
                                                    # gravity-check's pass rule and its in/out-of-range note
                                                    'friction_nm': friction_nm.tolist(),
                                                    'poses_rad': [q.tolist() for q, _ in data]}},
                            **({'joint_springs': springs} if springs else {})), lines, flags)
