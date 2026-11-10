#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

JOINTS = ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow']
TOPIC = '/velocity_controller/commands'
FREQ = 0.9           # rad/s  (angular frequency)
RAMP = 4.0           # s      (fade-in time)
START_DELAY = 0.05   # s
LIMIT = 8.0

# --- Range of motion -------------------------------------------------------
# Desired half-swing of the motion per joint (rad). Velocity amplitude is
# derived from this: v_peak = ROM_HALF * FREQ
ROM_HALF = {
    'left_shoulder_pitch': 0.55,
    'left_shoulder_roll':  0.45,
    'left_shoulder_yaw':   0.85,
    'left_elbow':          1.65,
}

# Hard position limits (rad) enforced from /joint_states feedback.
# SET THESE to your robot's safe limits.
LIMITS = {
    'left_shoulder_pitch': (-LIMIT, LIMIT),
    'left_shoulder_roll':  (-LIMIT, LIMIT),
    'left_shoulder_yaw':   (-LIMIT, LIMIT),
    'left_elbow':          (-LIMIT, LIMIT),
}
SOFT_MARGIN = 0.15   # rad: velocity tapers to zero over this distance before the limit


class CSVWaveController(Node):

    def __init__(self):
        super().__init__('csv_wave_controller')
        self.start_time = None
        self.last_print = None
        self.actual_vel = {j: 0.0 for j in JOINTS}
        self.actual_pos = {j: 0.0 for j in JOINTS}
        self.cmd = {j: 0.0 for j in JOINTS}

        self.pub = self.create_publisher(Float64MultiArray, TOPIC, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)

        self.get_logger().info('CSV wave controller started, waiting for velocity controller...')

    def limit_velocity(self, joint, v):
        """Taper/block velocity as the joint nears or exceeds its position limit."""
        lo, hi = LIMITS[joint]
        p = self.actual_pos[joint]

        if v > 0.0:
            room = hi - p
        elif v < 0.0:
            room = p - lo
        else:
            return 0.0

        if room <= 0.0:
            return 0.0                      # at/over limit: don't push further out
        if room < SOFT_MARGIN:
            return v * (room / SOFT_MARGIN)  # linear taper
        return v

    def publish_cmd(self, values):
        out = Float64MultiArray()
        for joint, v in zip(JOINTS, values):
            v = self.limit_velocity(joint, v)
            self.cmd[joint] = v
            out.data.append(v)
        self.pub.publish(out)

    def on_joint_state(self, msg: JointState):
        now = self.get_clock().now()

        if now.nanoseconds == 0:
            return

        for i, name in enumerate(msg.name):
            if name in self.actual_vel:
                if i < len(msg.velocity):
                    self.actual_vel[name] = msg.velocity[i]
                if i < len(msg.position):
                    self.actual_pos[name] = msg.position[i]

        if self.start_time is None:
            if self.pub.get_subscription_count() == 0:
                return
            self.start_time = now
            self.last_print = now
            self.get_logger().info('Velocity controller detected, starting ramp')

        t = (now - self.start_time).nanoseconds / 1e9 - START_DELAY

        if t < 0.0:
            self.publish_cmd([0.0] * len(JOINTS))
        else:
            x = min(1.0, t / RAMP)
            ramp = 0.5 * (1.0 - math.cos(math.pi * x))
            self.publish_cmd([
                ramp * (ROM_HALF[j] * FREQ) * math.cos(FREQ * t - idx)
                for idx, j in enumerate(JOINTS)
            ])

        if (now - self.last_print).nanoseconds / 1e9 >= 1.0:
            self.last_print = now
            lines = ['\n[CSV Diagnostics]']
            for j in JOINTS:
                lines.append('[%s] Pos: %+6.3f | Cmd Vel: %+6.3f | Act Vel: %+6.3f'
                             % (j, self.actual_pos[j], self.cmd[j], self.actual_vel[j]))
            self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = CSVWaveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()