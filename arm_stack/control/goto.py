"""`arm goto` (joint angles) and `arm goto-xyz` (tool position, IK): plan a goal,
check it, move there with move_to (Mode P, gravity offset) and leave the controller
holding it.

goto-xyz seeds IK with the current joint angles so it lands on the nearest solution
(no elbow flip), position only (mask='position'), and rejects a goal that did not
converge, is past a soft limit, or is inside a keep-out zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import time

from ..calib.common import clear_of_keep_out
from ..hw.client import Command
from .motion import move_to

DEG = np.pi / 180.0
GOTO_SPEED = 0.25          # rad/s peak, the rest-move speed
MAX_IK_ERR = 1e-3          # m: refuse a goal IK could not reach within 1 mm


@dataclass
class Plan:
    q: Optional[np.ndarray]    # goal, None if rejected
    reason: str = ''           # why it was rejected
    pos_err: float = 0.0       # m (goto-xyz)


def check_goal(model, zones, q) -> str:
    """'' if q is a goal we may move to, else why not."""
    bad = [i for i in range(len(q)) if not model.lower[i] <= q[i] <= model.upper[i]]
    if bad:
        return 'past the soft limits on ' + ', '.join(
            '%s (%.1f deg, limits %.1f..%.1f)' % (model.joint_names[i], q[i] / DEG, model.lower[i] / DEG,
                                                   model.upper[i] / DEG) for i in bad)
    if not clear_of_keep_out(zones, q, margin=0.0):
        return 'inside a keep-out zone'
    return ''


def plan_joints(model, zones, q_deg: Sequence[float]) -> Plan:
    q = np.asarray(q_deg, float) * DEG
    if q.shape != (len(model.lower),):
        return Plan(None, 'need %d joint angles, got %d' % (len(model.lower), q.size))
    why = check_goal(model, zones, q)
    return Plan(None if why else q, why)


WRIST = slice(4, None)       # forearm_yaw, wrist_pitch, wrist_yaw


def solve_wrist_locked(model, target, q_seed):
    lo, hi = model.lower.copy(), model.upper.copy()
    try:
        model.lower[WRIST] = np.clip(q_seed[WRIST], lo[WRIST], hi[WRIST])
        model.upper[WRIST] = model.lower[WRIST]
        return model.ik(target, q_seed, method='lm', mask='position')
    finally:
        model.lower[:] = lo
        model.upper[:] = hi


def plan_xyz(model, zones, xyz: Sequence[float], q_seed) -> Plan:
    target = np.asarray(xyz, float)
    if target.shape != (3,):
        return Plan(None, 'need X Y Z (m), got %d numbers' % target.size)
    q_seed = np.asarray(q_seed, float)
    # Wrist locked where it is (shoulder + elbow are 4 DOF for a 3D target, still
    # redundant): J7 is ~62 deg off the forearm, so a free wrist is the solver's
    # cheapest way to move the tool and it swings it tens of degrees for a few cm.
    # Only if that cannot reach does the whole arm solve.
    r = solve_wrist_locked(model, target, q_seed)
    if not r.success or r.pos_err > MAX_IK_ERR:
        r = model.ik(target, q_seed, method='lm', mask='position')
    if not r.success or r.pos_err > MAX_IK_ERR:
        return Plan(None, 'IK did not converge (%s, %.1f mm off): out of reach?' % (r.status, r.pos_err * 1e3),
                    r.pos_err)
    why = check_goal(model, zones, r.q)
    return Plan(None if why else r.q, why, r.pos_err)


def run(model, zones, client, plan: Plan, names, speed=GOTO_SPEED) -> int:
    s = client.state()
    if plan.q is None:
        print('goto rejected: %s' % plan.reason)
        return 1
    print('goto at %.2f rad/s:' % speed)
    for n, a, b in zip(names, s.q, plan.q):
        print('  %-15s %+7.1f -> %+7.1f deg' % (n, a / DEG, b / DEG))
    print('  tool           %s -> %s m' % (np.round(model.tool_position(s.q), 3),
                                          np.round(model.tool_position(plan.q), 3)))
    try:
        if hasattr(client, 'goto'):
            # the controller plans and steps the path itself (smooth whatever the
            # network timing); we only wait for it and watch
            duration = client.goto(plan.q, speed)
            t_end = time.monotonic() + duration + 0.5
            while time.monotonic() < t_end:
                s = client.step(0.05)
        else:
            s = move_to(client, plan.q, max_speed=speed, settle=0.5)
    except KeyboardInterrupt:
        # a streamed command newer than the goto takes over; it then goes stale and
        # the controller holds where the arm is
        client.send(Command(mode='P', q=client.state().q))
        print('goto stopped; the controller holds where it is.')
        return 130
    err = np.max(np.abs(s.q - plan.q)) / DEG
    print('there: tool %s m, worst joint %.2f deg off; the controller keeps holding.'
          % (np.round(model.tool_position(s.q), 3), err))
    return 0
