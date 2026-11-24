"""`arm wave`: raise the hand and wave, then go back to rest.

Pose from the model (v16 calibration, 2026-11-24): J2 117 deg (upper arm out to the side,
~27 deg above horizontal) and elbow 60 deg put the forearm vertical, hand ~0.48 m above
the shoulder, at J1 -6 deg. The wave rocks the elbow +-15 deg: the hand moves toward and
away from the head (+-9 cm sideways, ~1 cm forward/back). J1 would rock it forward/back
instead. The pose is nearly balanced (every joint under 0.5 Nm of gravity).

Route: one goto path (the controller runs it; soft limits and keep-out apply): rest ->
first wave extreme (a straight line for J1/J2, so the hand sweeps out to the side,
forward and up; J2 is past the keep-out gate by the time J1 has turned 8 deg), the
swings, then back to rest. The elbow keeps going through the raise into the first
swing and out of the last one into the lowering; at each swing end it turns without
a dwell. If the arm starts where the straight line would cross a keep-out zone, it
goes out to the side first (a separate move).
"""

from __future__ import annotations

import time

import numpy as np

from ..calib.common import clear_of_keep_out
from ..hw.client import Command
from .motion import move_to

DEG = np.pi / 180.0
SIDE = np.array([90, 117, 0, 60, 0, 0, 0]) * DEG     # arm out to the side, forearm forward
UP = np.array([-6, 117, 0, 60, 0, 0, 0]) * DEG       # forearm vertical: the waving pose
WAVE_JOINT = 3                                       # elbow
AMPLITUDE = 15 * DEG                                 # each way
MOVE_SPEED = 0.5                                     # rad/s peak, into and out of the pose


def _go(client, q, speed):
    if hasattr(client, 'goto'):
        duration = client.goto(q, speed)
        t_end = time.monotonic() + duration
        while time.monotonic() < t_end:
            client.step(0.02)
    else:
        move_to(client, q, max_speed=speed, min_time=0.3, settle=0.0)


def _line_clear(zones, a, b, steps=50):
    return all(clear_of_keep_out(zones, a + s * (b - a), margin=0.0) for s in np.linspace(0, 1, steps))


def run(client, home, zones=(), waves=3, swing_s=0.6):
    """swing_s: seconds per side-to-side swing (2 x AMPLITUDE). Ends at home."""
    speed = 1.875 * 2 * AMPLITUDE / swing_s            # min-jerk: duration = 1.875 dist / peak
    left, right = UP.copy(), UP.copy()
    left[WAVE_JOINT] -= AMPLITUDE
    right[WAVE_JOINT] += AMPLITUDE
    try:
        q = client.state().q
        if not _line_clear(zones, q, left) or not _line_clear(zones, left, home):
            print('going out to the side first (%.1f rad/s)' % MOVE_SPEED)
            _go(client, SIDE, MOVE_SPEED)
        print('raise, %d waves of +-%.0f deg on the elbow at %.1f s per swing, back to rest'
              % (waves, AMPLITUDE / DEG, swing_s))
        path = [(left, MOVE_SPEED)] + [(p, speed) for _ in range(waves) for p in (right, left)] \
            + [(home, MOVE_SPEED)]
        if hasattr(client, 'goto_path'):
            duration = client.goto_path([p for p, _ in path], [v for _, v in path])
            t_end = time.monotonic() + duration + 0.3
            while time.monotonic() < t_end:
                client.step(0.02)
        else:
            for p, v in path:
                _go(client, p, v)
        print('At rest; the controller keeps holding.')
        return 0
    except KeyboardInterrupt:
        client.send(Command(mode='P', q=client.state().q))
        print('wave stopped; the controller holds where it is.')
        return 130
