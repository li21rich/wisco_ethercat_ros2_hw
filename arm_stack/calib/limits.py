"""Stage 3, `arm limits`: find the hard stops, set soft limits 3 deg inside.

A joint with `hard_stops: false` in the arm config (no physical stop, only cables)
is not driven past its URDF range: its soft limits are the URDF range, and the
sweep below still checks it reaches both ends.

Each joint moves slowly toward each URDF limit and on past it in Mode P, with
gravity feedforward and 0x6072 clamped to the model gravity torque plus a
margin. Confirm contact from sustained low speed and measured torque in the
approach direction above gravity. Then sweep to the inset software limits.
Contact detection is adapted from Enactic's OpenArm zero-position calibration.
"""

from __future__ import annotations

import numpy as np

from ..control.motion import move_to, stream
from .common import DEG, StageContext, StageResult, safe_home

INSET = 3 * DEG
SEARCH_BEYOND = 10 * DEG
SPEED = 0.15            # rad/s
STALL_ERROR = 1.5 * DEG
HOLD_STILL = 1.0 * DEG      # a held joint that moves more than this is not at a stop (yet)
HOLD_TIMEOUT_S = 2.0        # held this long without settling or catching up: abort
RUNAWAY_SETTLE_S = 0.3      # a fast search may pass 2 x STALL_ERROR still decelerating into the stop
BREAKAWAY_STUCK = 8 * DEG   # lag at which a joint that never broke free is at a stop


def path_gravity(model, home, j, end, samples=25):
    """Largest |gravity torque| per joint while joint j sweeps from home to `end`."""
    q = home.copy()
    worst = np.zeros(model.n)
    for v in np.linspace(home[j], end, samples):
        q[j] = v
        worst = np.maximum(worst, np.abs(model.gravity(q)))
    return worst


def _reason(trace, why, s, j, lag):
    if trace is not None:
        trace['reason'] = '%s: q %.2f deg, lag %.2f deg, qd %.3f rad/s, tau %.2f Nm' % (
            why, s.q[j] / DEG, lag / DEG, s.qd[j], s.tau[j])


def find_stop(ctx, j, direction, home, clamp, reference=None, *, approach=8 * DEG,
              beyond=SEARCH_BEYOND, speed=None, approach_speed=0.4, baseline='model',
              clip_near=True, breakaway=0.0, trace=None, confirm_s=0.04, **move_kw):
    """Search for joint j's stop in `direction`; the stop angle or None.

    approach: start the slow search this far before `reference`. beyond: search
    this far past it. baseline: 'model' compares the pushing torque with modeled
    gravity; 'measured' with the joint's own torque while it moved freely just
    before (for uncalibrated coordinates, where modeled gravity is unreliable).
    clip_near=False lets the search start outside the model limits (a stop beyond
    the URDF range). breakaway: a search that starts from rest ignores contacts until the
    joint has moved this far (stiction breakaway is not a stop); a joint that never breaks
    free (lag past BREAKAWAY_STUCK) is at a stop where it started. trace: a dict that gets
    per-step 'rows' (t, q_cmd, q, qd, tau, tau_cmd, lag, phase) and the accepting 'reason'.
    confirm_s: how long a contact must persist (a joint that sticks and slips is not a stop).
    move_kw go to every command (e.g. gravity_ff=False)."""
    client, model = ctx.client, ctx.model
    if speed is None:
        speed = SPEED * (2 if ctx.quick else 1)
    if baseline not in ('model', 'measured'):
        raise ValueError('baseline must be model or measured')
    if reference is None:
        reference = model.upper[j] if direction > 0 else model.lower[j]
    target_end = reference + direction * beyond
    # approach to just inside the expected stop at normal speed
    near = home.copy()
    near[j] = reference - direction * approach
    if clip_near:
        near[j] = np.clip(near[j], model.lower[j], model.upper[j])
    move_kw = dict(move_kw)
    move_kw.pop('ignore_soft_limits', None)
    # an unclipped start may lie past the soft limits, like the search itself
    approach_kw = dict(move_kw, ignore_soft_limits=not clip_near)
    move_to(client, near, max_speed=approach_speed, torque_clamp=clamp, **approach_kw)
    s = client.state()
    q_cmd = s.q.copy()
    q_start = float(s.q[j])
    n = int(abs(target_end - q_cmd[j]) / speed / client.cmd_dt) + 1
    stall_q = None
    contact_since = None
    # Scale contact detection to the existing search torque margin for this joint.
    # Subtract gravity: a stationary, loaded shoulder already draws current.
    contact_nm = 0.04 * ctx.arm.joints[j].peak_torque_nm
    free_tau = None
    alpha = min(1.0, client.cmd_dt / 0.2)      # 0.2 s running mean of the free-motion torque
    runaway_since = None
    steps = 0
    while steps < n:
        # held iterations (runaway_since set) do not use up the travel
        if runaway_since is None:
            q_cmd[j] += direction * speed * client.cmd_dt
            steps += 1
        qd = np.zeros(model.n)
        qd[j] = 0.0 if runaway_since is not None else direction * speed
        s = stream(client, [q_cmd.copy()], [qd], torque_clamp=clamp, ignore_soft_limits=True, **move_kw)
        lag = direction * (q_cmd[j] - s.q[j])
        if trace is not None:
            tau_cmd = getattr(s, 'tau_cmd', None)
            trace.setdefault('rows', []).append((
                float(s.t), float(q_cmd[j]), float(s.q[j]), float(s.qd[j]), float(s.tau[j]),
                float(tau_cmd[j]) if tau_cmd is not None else float('nan'), float(lag),
                'hold' if runaway_since is not None else 'move'))
        if breakaway > 0 and direction * (s.q[j] - q_start) < breakaway:
            # still breaking away from rest: no contact yet, unless it never moves at all
            if lag > BREAKAWAY_STUCK:
                stall_q = float(s.q[j])
                _reason(trace, 'never broke free (stuck at start)', s, j, lag)
                break
            continue
        if baseline == 'measured':
            if free_tau is None:
                free_tau = float(s.tau[j])
            elif lag < 0.5 * STALL_ERROR:
                free_tau += alpha * (float(s.tau[j]) - free_tau)
            reference_tau = free_tau
        else:
            reference_tau = model.gravity(s.q)[j]
        # stopped, behind the command, and pushing in the approach direction above gravity
        pressing = (lag > STALL_ERROR and abs(s.qd[j]) < 0.02
                    and direction * (s.tau[j] - reference_tau) >= contact_nm)
        if runaway_since is not None or lag > 2 * STALL_ERROR:
            # The command runs away from a joint that has stopped: hold it (latched). A joint
            # that catches up again was only sticking (stick-slip) and the search resumes; one
            # that stays put (within HOLD_STILL) for confirm_s is at its stop. A fast search
            # reaches the lag bound still decelerating into the stop, and a stop gives a
            # little, so the stillness timer restarts whenever the joint moves.
            if runaway_since is None:
                runaway_since = s.t
                still_q, still_since = float(s.q[j]), s.t
            if abs(s.q[j] - still_q) > HOLD_STILL:
                still_q, still_since = float(s.q[j]), s.t
            still = s.t - still_since
            if lag < 0.5 * STALL_ERROR:
                runaway_since = None          # caught up with the held command: keep searching
                continue
            if pressing and still >= confirm_s:
                stall_q = float(s.q[j])
                _reason(trace, 'pressing, still for %.2f s' % still, s, j, lag)
                break
            # stopped short of the held command after moving freely: the stop (a joint that
            # never moved, without any torque, is a dead drive, not a stop)
            travelled = direction * (s.q[j] - q_start) > 2 * STALL_ERROR
            if travelled and abs(s.qd[j]) < 0.02 and still >= max(confirm_s, RUNAWAY_SETTLE_S):
                stall_q = float(s.q[j])
                _reason(trace, 'still short of the held command for %.2f s' % still, s, j, lag)
                break
            if s.t - runaway_since < HOLD_TIMEOUT_S:
                continue
            print(f'{ctx.arm.names[j]}: search stopped at {s.q[j]/DEG:.2f} deg, '
                  f'lag {lag/DEG:.2f} deg, speed {s.qd[j]:.3f} rad/s, torque {s.tau[j]:.2f} Nm')
            break
        if pressing:
            if contact_since is None:
                contact_since = s.t
            if s.t - contact_since >= confirm_s:
                stall_q = float(s.q[j])
                _reason(trace, 'pressing for %.2f s' % (s.t - contact_since), s, j, lag)
                break
            continue
        contact_since = None
    # back off
    move_to(client, near, max_speed=min(0.3, approach_speed), torque_clamp=clamp, **approach_kw)
    return stall_q


def run(ctx: StageContext) -> StageResult:
    client, model, arm = ctx.client, ctx.model, ctx.arm
    home = safe_home(model)
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    lines, flags = [], []
    soft = {}
    stops = {}
    ok = True
    for j, name in enumerate(arm.names):
        if not arm.joints[j].hard_stops:
            # No physical stop to find (config `hard_stops: false`): driving past the
            # URDF range would only meet the cables. The URDF range is the soft limit.
            soft[name] = [float(model.lower[j]), float(model.upper[j])]
            stops[name] = None
            lines.append('%-15s no hard stops (hard_stops: false) -> soft = URDF [%+7.2f, %+7.2f] deg'
                         % (name, model.lower[j] / DEG, model.upper[j] / DEG))
            flags.append('%s: no stop search (hard_stops: false); soft limits are the URDF range, '
                         'check the cable routing reaches it' % name)
            continue
        found = []
        for direction in (-1, +1):
            end = (model.upper[j] + SEARCH_BEYOND) if direction > 0 else (model.lower[j] - SEARCH_BEYOND)
            # model gravity anywhere on the way, plus a margin (spec: 0x6072 = gravity + margin)
            g = path_gravity(model, home, j, end)
            clamp = g + 0.08 * peak
            max_t = np.array([jt.max_torque_nm for jt in arm.joints])
            if np.any(g > 0.9 * max_t):
                flags.append('%s: gravity on the way to its %s limit (%.1f Nm on %s) is near the torque limit'
                             % (name, 'upper' if direction > 0 else 'lower', g.max(),
                                arm.names[int(np.argmax(g / max_t))]))
            stop = find_stop(ctx, j, direction, home, clamp)
            if stop is None:
                flags.append('%s: no confirmed stop on bounded search (maximum %.0f deg past the %s limit)'
                             % (name, SEARCH_BEYOND / DEG, 'upper' if direction > 0 else 'lower'))
                # find_stop leaves the joint 8 deg inside a limit, maybe in a high-gravity
                # pose: go home before giving up. Nothing is written for an unconfirmed stop.
                move_to(client, home, max_speed=0.3, torque_clamp=clamp)
                return StageResult('limits', False, {'hard_stops_rad': stops}, {}, lines, flags)
            found.append(stop)
        lo, hi = found
        soft[name] = [float(lo + INSET), float(hi - INSET)]
        stops[name] = [float(lo), float(hi)]
        lines.append('%-15s stops [%+7.2f, %+7.2f] deg (URDF [%+7.2f, %+7.2f]) -> soft [%+7.2f, %+7.2f]' % (
            name, lo / DEG, hi / DEG, model.lower[j] / DEG, model.upper[j] / DEG,
            soft[name][0] / DEG, soft[name][1] / DEG))
        if lo > model.lower[j] + 1 * DEG or hi < model.upper[j] - 1 * DEG:
            flags.append('%s: hard stop inside the URDF limit; update the URDF or the stop' % name)

    # verify: sweep to each soft limit, no stall on the way
    for j, name in enumerate(arm.names):
        for value in soft[name]:
            q = home.copy()
            q[j] = value
            clamp = path_gravity(model, home, j, value) + 0.15 * peak
            s = move_to(client, q, max_speed=0.6 if ctx.quick else 0.3, torque_clamp=clamp)
            if abs(s.q[j] - value) > STALL_ERROR:
                ok = False
                flags.append('%s: stalled at %.2f deg before its soft limit %.2f deg'
                             % (name, s.q[j] / DEG, value / DEG))
        move_to(client, home, max_speed=0.4)
    return StageResult('limits', ok, {'hard_stops_rad': stops}, {'soft_limits_rad': soft}, lines, flags)
