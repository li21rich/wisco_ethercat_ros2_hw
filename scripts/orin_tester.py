#!/usr/bin/env python3
"""Stand-in for the Orin: drive the arm by hand through the Orin-facing API.

Talks to the same topics the Orin will use (see arm_controller.hpp):
  /arm_controller/joint_position  CSP: joint angles (+ velocity) streamed at --rate
  /arm_controller/joint_torque    CST: joint torques on top of gravity compensation
  /arm_controller/joint_state     feedback: position, velocity, measured torque

  ros2 run wisco_ethercat_ros2_hw orin_tester.py [--rate 100] [--speed 20] [--max-torque 3] [--pulse 0.1]

Commands (angles in deg, joints by number 1-7 or name):
  p a1 a2 ... a7     CSP: move every joint to these angles (min-jerk at --speed)
  p J a              CSP: move one joint, the rest stay
  t J n [S]          CST pulse: joint J gets n Nm (pure torque, others 0, all on top of
                     gravity comp) for S s (default --pulse 0.1), then the controller
                     holds wherever the arm ended up
  t n1 ... n7 [S]    same, every joint
  tc J n / tc n1..n7 CST continuous: stays on until hold/rest (adds to the current set)
  speed DEG_S        CSP move speed (peak)
  s                  print the state
  hold               stop streaming: the controller holds where the arm is
  rest               stop streaming and move to the hanging rest pose (`arm rest`)
  q / Ctrl-D         rest, then quit (Ctrl-C quits holding where it is)
Continuous torque changes ramp over 0.5 s; everything is capped at --max-torque. Nothing is sent
until the first p or t, so starting this never moves the arm.
"""

import argparse
import os
import subprocess
import math
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

DEG = math.pi / 180.0
NS = '/arm_controller'


class Orin(Node):

    def __init__(self, rate, speed, max_torque, pulse):
        super().__init__('orin_tester')
        self.lock = threading.Lock()
        self.names = []
        self.q = self.qd = self.tau = None
        self.mode = 'idle'           # idle | p | t
        self.speed = speed * DEG
        self.max_torque = max_torque
        self.path = None             # (t0, T, q0, q1)
        self.goal = None
        self.tau_cmd = self.tau_from = None
        self.tau_t0 = 0.0
        self.pulse = pulse
        self.tau_end = None          # pulse end (monotonic), None = continuous
        self.ramp = 0.5
        self.pos_pub = self.create_publisher(JointState, NS + '/joint_position', 10)
        self.tor_pub = self.create_publisher(JointState, NS + '/joint_torque', 10)
        self.create_subscription(JointState, NS + '/joint_state', self.on_state, 10)
        self.create_timer(1.0 / rate, self.tick)

    def on_state(self, m):
        with self.lock:
            self.names = list(m.name)
            self.q, self.qd, self.tau = list(m.position), list(m.velocity), list(m.effort)

    # --- streaming ---------------------------------------------------------
    def tick(self):
        with self.lock:
            if self.mode == 'p' and self.path:
                t0, T, q0, q1 = self.path
                s = min((time.monotonic() - t0) / T, 1.0)
                b = s ** 3 * (10 - 15 * s + 6 * s * s)
                db = 30 * s * s * (1 - s) ** 2 / T if s < 1.0 else 0.0
                pos = [a + (c - a) * b for a, c in zip(q0, q1)]
                vel = [(c - a) * db for a, c in zip(q0, q1)]
                self.pos_pub.publish(JointState(name=self.names, position=pos, velocity=vel))
            elif self.mode == 't' and self.tau_end is not None and time.monotonic() >= self.tau_end:
                self.mode = 'idle'     # pulse over: stop streaming, the controller holds
            elif self.mode == 't':
                a = min((time.monotonic() - self.tau_t0) / self.ramp, 1.0) if self.ramp > 0 else 1.0
                eff = [f + (c - f) * a for f, c in zip(self.tau_from, self.tau_cmd)]
                self.tor_pub.publish(JointState(name=self.names, effort=eff))

    def _current_target(self):
        """Where the CSP stream is now (or the measured pose when not streaming CSP)."""
        if self.mode == 'p' and self.path:
            t0, T, q0, q1 = self.path
            s = min((time.monotonic() - t0) / T, 1.0)
            b = s ** 3 * (10 - 15 * s + 6 * s * s)
            return [a + (c - a) * b for a, c in zip(q0, q1)]
        return list(self.q)

    def _current_torque(self):
        if self.mode != 't':
            return [0.0] * len(self.names)
        a = min((time.monotonic() - self.tau_t0) / self.ramp, 1.0) if self.ramp > 0 else 1.0
        return [f + (c - f) * a for f, c in zip(self.tau_from, self.tau_cmd)]

    # --- commands ----------------------------------------------------------
    def joint(self, token):
        if token.isdigit() and 1 <= int(token) <= len(self.names):
            return int(token) - 1
        if token in self.names:
            return self.names.index(token)
        raise ValueError('no joint %r (1-%d or %s)' % (token, len(self.names), ', '.join(self.names)))

    def position(self, args):
        with self.lock:
            start = self._current_target()
            goal = list(self.goal) if self.mode == 'p' and self.goal else list(start)
            if len(args) == len(self.names):
                goal = [float(a) * DEG for a in args]
            elif len(args) == 2:
                goal[self.joint(args[0])] = float(args[1]) * DEG
            else:
                raise ValueError('p takes %d angles, or a joint and an angle' % len(self.names))
            dist = max(abs(g - s) for g, s in zip(goal, start))
            T = max(0.3, 1.875 * dist / self.speed)
            self.path = (time.monotonic(), T, start, goal)
            self.goal = goal
            self.mode = 'p'
        return 'CSP: %.1f s to [%s] deg' % (T, ', '.join('%.1f' % (g / DEG) for g in goal))

    def torque(self, args, continuous=False):
        with self.lock:
            n = len(self.names)
            dur = None
            if not continuous and len(args) in (3, n + 1):
                dur = float(args[-1])
                args = args[:-1]
            if dur is not None and not dur > 0:
                raise ValueError('pulse length must be > 0 s')
            if continuous and self.mode == 't' and self.tau_end is None:
                now, cmd = self._current_torque(), list(self.tau_cmd)
            else:
                now, cmd = [0.0] * n, [0.0] * n
            if len(args) == n:
                cmd = [float(a) for a in args]
            elif len(args) == 2:
                cmd[self.joint(args[0])] = float(args[1])
            else:
                raise ValueError('t takes a joint and a torque, or %d torques (then optional seconds)' % n)
            capped = [max(-self.max_torque, min(self.max_torque, c)) for c in cmd]
            note = '' if capped == cmd else ' (capped at %.1f Nm)' % self.max_torque
            t0 = time.monotonic()
            if continuous:
                self.tau_end, self.ramp = None, 0.5
            else:
                dur = dur or self.pulse
                self.tau_end, self.ramp = t0 + dur, 0.0
                now = [0.0] * n
            self.tau_from, self.tau_cmd, self.tau_t0 = now, capped, t0
            self.mode, self.path, self.goal = 't', None, None
        what = 'continuous' if continuous else '%.2f s pulse' % dur
        return 'CST %s: [%s] Nm + gravity comp%s' % (what, ', '.join('%.2f' % c for c in capped), note)

    def hold(self):
        with self.lock:
            self.mode, self.path, self.goal = 'idle', None, None
        return 'stopped streaming: the controller holds the arm where it is'

    def state(self):
        with self.lock:
            rows = ['%-3s %-22s %8s %9s %8s' % ('#', 'joint', 'deg', 'deg/s', 'Nm')]
            for i, n in enumerate(self.names):
                rows.append('%-3d %-22s %8.2f %9.2f %8.2f'
                            % (i + 1, n, self.q[i] / DEG, self.qd[i] / DEG, self.tau[i]))
            rows.append('streaming: %s' % {'idle': 'nothing (controller holds)',
                                           'p': 'CSP', 't': 'CST'}[self.mode])
        return '\n'.join(rows)


def go_rest(node):
    """Hand over to `arm rest` (keep-out aware goto to J1 +90, rest 0)."""
    print(node.hold())
    time.sleep(0.1)
    from ament_index_python.packages import get_package_share_directory
    arm = os.path.join(get_package_share_directory('wisco_ethercat_ros2_hw'), 'config', 'arms', 'arm01.yaml')
    try:
        subprocess.run(['ros2', 'run', 'wisco_ethercat_ros2_hw', 'arm', 'rest', '--no-log', '--arm', arm])
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rate', type=float, default=100.0, help='stream rate, Hz (default 100)')
    ap.add_argument('--speed', type=float, default=20.0, help='CSP move peak speed, deg/s (default 20)')
    ap.add_argument('--max-torque', type=float, default=3.0, help='CST cap per joint, Nm (default 3)')
    ap.add_argument('--pulse', type=float, default=0.1, help='t pulse length, s (default 0.1)')
    a = ap.parse_args()

    rclpy.init()
    node = Orin(a.rate, a.speed, a.max_torque, a.pulse)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    t_end = time.monotonic() + 5.0
    while node.q is None and time.monotonic() < t_end:
        time.sleep(0.05)
    if node.q is None:
        print('no %s/joint_state: is the arm launched?' % NS)
        executor.shutdown()
        rclpy.try_shutdown()
        return 1
    print(node.state())
    print("type 'help' for commands")
    try:
        while True:
            try:
                line = input('orin> ').split()
            except EOFError:
                print()
                break
            if not line:
                continue
            cmd, args = line[0], line[1:]
            try:
                if cmd == 'p':
                    print(node.position(args))
                elif cmd == 't':
                    print(node.torque(args))
                elif cmd == 'tc':
                    print(node.torque(args, continuous=True))
                elif cmd == 'speed':
                    node.speed = float(args[0]) * DEG
                    print('CSP speed %.1f deg/s' % (node.speed / DEG))
                elif cmd == 's':
                    print(node.state())
                elif cmd == 'hold':
                    print(node.hold())
                elif cmd == 'rest':
                    go_rest(node)
                elif cmd in ('q', 'quit', 'exit'):
                    break
                elif cmd in ('help', 'h', '?'):
                    print(__doc__.split('Commands')[1])
                else:
                    print("unknown command %r; type 'help'" % cmd)
            except (ValueError, IndexError) as e:
                print(e)
        go_rest(node)
    except KeyboardInterrupt:
        print()
        print(node.hold())
    time.sleep(0.1)
    executor.shutdown()
    spin.join(1.0)
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
