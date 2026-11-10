#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# per-joint characterization overrides (Manually tune values HERE)
JOINT_CHARACTERIZATION_OVERRIDES = {
    'joint_1': {'k_stiction': 1.8},
    'joint_2': {'k_stiction': 0.85},
    'joint_3': {'k_stiction': 1.6},
    'joint_4': {'k_stiction': 1.9},
}

# defaults (Do not overwrite)
ACTUATOR_PROFILES = {
    'x4_10':  {'k_stiction': 0.35, 'torque_limit': 2.5},
    'x4_36':  {'k_stiction': 0.90, 'torque_limit': 6.0},
    'x6_60':  {'k_stiction': 1.50, 'torque_limit': 12.0},
    'x8_120': {'k_stiction': 2.80, 'torque_limit': 25.0},
}


def _merge_profile(model, joint):
    """Return a per-joint profile: Tier 1 model defaults merged with Tier 2 overrides."""
    base = dict(ACTUATOR_PROFILES.get(model, ACTUATOR_PROFILES['x4_10']))
    base.update(JOINT_CHARACTERIZATION_OVERRIDES.get(joint, {}))
    return base


class MultiActuatorTorqueController(Node):

    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        # ----------------------- ROS parameters -----------------------
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4'])
        self.declare_parameter('actuator_chain', ['x4_10', 'x4_10', 'x4_36', 'x4_36'])
        self.declare_parameter('topic_names', ['/effort_controller/commands'])
        self.declare_parameter('speed_factor', 1.5)
        self.declare_parameter('amplitude', 1.0)
        self.declare_parameter('phase_offset', 0.4)

        self.joint_names = list(self.get_parameter('joint_names').value)
        chain = list(self.get_parameter('actuator_chain').value) or ['x4_10']
        chain += [chain[-1]] * max(0, len(self.joint_names) - len(chain))
        self.actuator_chain = chain

        self.topic_names = list(self.get_parameter('topic_names').value)
        self.speed_factor = float(self.get_parameter('speed_factor').value)
        self.amplitude = float(self.get_parameter('amplitude').value)
        self.phase_offset = float(self.get_parameter('phase_offset').value)

        # Map joint -> actuator model
        self.joint_model = dict(zip(self.joint_names, self.actuator_chain))

        # Build per-joint gain profiles
        self.profiles = {
            joint: _merge_profile(self.joint_model.get(joint, 'x4_10'), joint)
            for joint in self.joint_names
        }

        # ----------------------- timing / state -----------------------
        self.start_time = self.get_clock().now()
        self.last_print_time = self.get_clock().now()

        self.vel_alpha = 0.35
        self.current_cmd = {joint: 0.0 for joint in self.joint_names}
        self.latest_states = {
            joint: {'velocity': 0.0, 'raw_velocity': 0.0, 'effort': 0.0}
            for joint in self.joint_names
        }

        self._last_target_v = {joint: 0.0 for joint in self.joint_names}
        self._last_ff = {joint: 0.0 for joint in self.joint_names}

        # ----------------------- publishers & subscriber --------------
        self.cmd_pubs = [
            self.create_publisher(Float64MultiArray, topic, 10)
            for topic in self.topic_names
        ]
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.get_logger().info(
            'MultiActuatorTorqueController initialized (pure feedforward wave mode) | joints: %s | models: %s' %
            (', '.join(self.joint_names), ', '.join(self.actuator_chain)))

    def joint_state_callback(self, msg: JointState):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        for i, name in enumerate(msg.name):
            if name in self.latest_states:
                raw_vel = msg.velocity[i] if i < len(msg.velocity) else self.latest_states[name]['raw_velocity']
                self.latest_states[name]['raw_velocity'] = raw_vel

                prev_vel = self.latest_states[name]['velocity']
                self.latest_states[name]['velocity'] = ((1.0 - self.vel_alpha) * prev_vel +
                                                        self.vel_alpha * raw_vel)
                if i < len(msg.effort):
                    self.latest_states[name]['effort'] = msg.effort[i]

        cmd_array = [0.0] * len(self.joint_names)
        self._run_pure_ff_control(t, cmd_array)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = cmd_array
        for pub in self.cmd_pubs:
            pub.publish(cmd_msg)

        if (now - self.last_print_time).nanoseconds / 1e9 >= 1.0:
            self.last_print_time = now
            self._log_diagnostics()

    def _run_pure_ff_control(self, t, cmd_array):
        ramp_scale = min(1.0, t / 2.0)

        for idx, joint in enumerate(self.joint_names):
            prof = self.profiles[joint]

            target_v = ramp_scale * self.amplitude * math.cos(
                self.speed_factor * t - idx * self.phase_offset)
            self._last_target_v[joint] = target_v

            safe_stiction = min(prof['k_stiction'], prof['torque_limit'] * 0.4)
            raw_cmd = safe_stiction * math.tanh(target_v * 2)
            self._last_ff[joint] = raw_cmd

            tl = prof['torque_limit']
            clamped_cmd = max(-tl, min(tl, raw_cmd))

            self.current_cmd[joint] = clamped_cmd
            cmd_array[idx] = clamped_cmd

    def _log_diagnostics(self):
        lines = ['\n[Pure FF Torque Diagnostics]']
        for joint in self.joint_names:
            model = self.joint_model.get(joint, '?')
            prof = self.profiles[joint]
            vel = self.latest_states[joint]['velocity']
            cmd = self.current_cmd[joint]
            lines.append(
                '[%s | %s] Tgt Vel: %+6.3f | Act Vel: %+6.3f | Cmd Eff: %+6.3f / %+6.3f N.m (FF Torque)' %
                (joint, model,
                 self._last_target_v[joint], vel,
                 cmd, prof['torque_limit']))
        self.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = MultiActuatorTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()