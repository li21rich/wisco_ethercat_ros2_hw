"""Stage 6, `arm friction`: Coulomb plus viscous per joint.

Each joint runs constant-velocity sweeps in both directions at 5 speeds
(others held) in Mode P. Over the constant-speed part of each sweep, the
friction torque is the measured torque minus the model gravity (stage 5) and
inertia (zero at constant speed). For each joint

    f(v) = Fc sign(v) + Fv v + b

is fit to the 10 points, where b soaks up any leftover gravity error that is
the same in both directions. Pass: R^2 above 0.9.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import Trace, constant_velocity, move_to
from .common import DEG, StageContext, StageResult, r_squared, safe_home

R2_MIN = 0.9
SPEED_FRACTIONS = (0.05, 0.1, 0.2, 0.35, 0.5)
SPAN = 30 * DEG
GATE_CLEARANCE = 5 * DEG   # past a keep-out gate's gate_min while its joint sweeps


def fit_friction(vs, fs):
    """f(v) = Fc sign(v) + Fv v + b with Fc, Fv >= 0. A negative free-fit Fv is the
    Stribeck dip (friction falling off the low-speed end, arm01 J3/J4/J7 2026-11-23),
    which this model cannot express: refit with Fv = 0. Returns (fc, fv, bias, r2)."""
    vs, fs = np.asarray(vs, float), np.asarray(fs, float)
    A = np.column_stack([np.sign(vs), vs, np.ones_like(vs)])
    coef, *_ = np.linalg.lstsq(A, fs, rcond=None)
    if coef[1] < 0:
        c2, *_ = np.linalg.lstsq(A[:, [0, 2]], fs, rcond=None)
        coef = np.array([c2[0], 0.0, c2[1]])
    fc, fv, bias = (float(c) for c in coef)
    return fc, fv, bias, r_squared(fs, A @ coef)


def run(ctx: StageContext) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    home = safe_home(model)
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    lines, flags, out, metrics = [], [], {}, {}
    ok = True
    for j, spec in enumerate(arm.joints):
        # a keep-out joint sweeps with its gate open (arm01: J2 rolled out before J1
        # swings past 60..120 deg), or the control law would hold it at the band edge
        base = home.copy()
        for joint, gate, _lo, _hi, gate_min in getattr(arm, 'keep_out', ()):
            if joint == j:
                base[gate] = min(max(base[gate], gate_min + GATE_CLEARANCE), model.upper[gate] - 5 * DEG)
        if np.any(base != home):
            move_to(client, base, max_speed=0.4)
        mid = home[j]
        vmax = min(spec.max_vel_rad_s, 1.5)
        speeds = [f * vmax for f in SPEED_FRACTIONS]
        vs, fs = [], []
        for v in speeds:
            # long enough for the drive to settle at speed (>= 1.5 s) where the range allows
            span = min(max(SPAN, v * 1.5), model.upper[j] - model.lower[j] - 10 * DEG)
            lo = max(model.lower[j] + 5 * DEG, mid - span / 2)
            hi = min(model.upper[j] - 5 * DEG, lo + span)
            lo = max(model.lower[j] + 5 * DEG, hi - span)
            for a, b in ((lo, hi), (hi, lo)):
                tr = Trace()
                q_start = base.copy()
                q_start[j] = a
                clamp = np.minimum(peak, np.abs(model.gravity(q_start)) * 1.5 + 0.3 * peak)
                constant_velocity(client, j, a, b, v, trace=tr, torque_clamp=clamp)
                d = tr.arrays()
                # constant-speed window: skip the start-up transient and the end
                n = len(d['t'])
                w = slice(int(0.35 * n), int(0.9 * n))
                tau = d['tau'][w, j]
                g = np.array([model.gravity(q)[j] for q in d['q'][w]])
                # the drive may still be settling inside the window: take out the mean
                # inertial torque M_jj * qdd, qdd = least-squares slope of qd over the window
                tw = d['t'][w] - d['t'][w].mean()
                qdd_mean = float(np.dot(tw, d['qd'][w, j] - d['qd'][w, j].mean()) / max(np.dot(tw, tw), 1e-12))
                M_jj = model.mass_matrix(d['q'][n // 2])[j, j]
                vs.append(float(np.mean(d['qd'][w, j])))
                fs.append(float(np.mean(tau - g) - M_jj * qdd_mean))
        vs, fs = np.array(vs), np.array(fs)
        fc, fv, bias, r2 = fit_friction(vs, fs)
        good = r2 > R2_MIN and fc >= 0
        ok &= good
        out[spec.name] = {'coulomb_nm': float(max(fc, 0.0)), 'viscous_nm_s_per_rad': float(max(fv, 0.0))}
        metrics[spec.name] = {'r2': r2, 'bias_nm': float(bias), 'speeds': vs.tolist(), 'friction': fs.tolist()}
        lines.append('%-15s Fc %.3f Nm, Fv %.3f Nm s/rad, R^2 %.3f, offset %+.3f Nm%s' % (
            spec.name, fc, fv, r2, bias, '' if good else '  <-- FAIL'))
        if abs(bias) > 0.1 * max(abs(fc), 0.05):
            flags.append('%s: %.3f Nm offset common to both directions; gravity fit error along this sweep'
                         % (spec.name, bias))
        move_to(client, home, max_speed=0.4)
    return StageResult('friction', ok, metrics, {'friction': out}, lines, flags)
