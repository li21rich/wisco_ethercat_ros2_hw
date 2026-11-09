#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# Tier 1: model-level gain / safety defaults (baseline values per actuator model)
# ---------------------------------------------------------------------------
ACTUATOR_PROFILES = {
    'x4_10':  {'k_stiction': 0.35, 'k_ff': 0.15, 'torque_limit': 2.5},
    'x4_36':  {'k_stiction': 0.90, 'k_ff': 0.50, 'torque_limit': 6.0},
    'x6_60':  {'k_stiction': 1.50, 'k_ff': 0.80, 'torque_limit': 12.0},
    'x8_120': {'k_stiction': 2.80, 'k_ff': 1.50, 'torque_limit': 25.0},
}

# ---------------------------------------------------------------------------
# Tier 2: per-joint characterization overrides (from your calibration run)
# ADJUST MANUALLY AFTER PASTING!
# ---------------------------------------------------------------------------
JOINT_CHARACTERIZATION_OVERRIDES = { 

   'joint_1': {'k_stiction': 1.025, 'k_ff': 0.043},
   'joint_2': {'k_stiction': 1.4921, 'k_ff': 0.013},
   'joint_3': {'k_stiction': 1.2639, 'k_ff': 0.1983},
   'joint_4': {'k_stiction': 2.2876, 'k_ff': 0.2987},
}

def _merge_profile(model, joint):
    """Return a per-joint profile: Tier 1 model defaults merged with Tier 2 overrides."""
    base = dict(ACTUATOR_PROFILES.get(model, ACTUATOR_PROFILES['x4_10']))
    base.update(JOINT_CHARACTERIZATION_OVERRIDES.get(joint, {}))
    return base


class MultiActuatorTorqueController(Node):
    """
    Pure feedforward multi-actuator effort controller (Option A):
    - Eliminates velocity P/I feedback loops that fight torque commands.
    - Uses model-based viscous feedforward + smoothed Coulomb stiction breakaway.
    - Safe per-joint torque clamping.
    """

    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        # ----------------------- ROS parameters -----------------------
        self.declare_parameter('joint_names', [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8',
            'joint_9', 'joint_10', 'joint_11', 'joint_12', 'joint_13', 'joint_14', 'joint_15', 'joint_16'])
        self.declare_parameter('actuator_chain', ['x4_10'])
        self.declare_parameter('topic_names', ['/effort_controller/commands'])
        self.declare_parameter('characterize_mode', False)
        self.declare_parameter('speed_factor', 1.5)
        self.declare_parameter('amplitude', 1.0)
        self.declare_parameter('phase_offset', 0.4)

        self.joint_names = self.get_parameter('joint_names').get_parameter_value().string_array_value
        self.actuator_chain = self.get_parameter('actuator_chain').get_parameter_value().string_array_value
        self.topic_names = self.get_parameter('topic_names').get_parameter_value().string_array_value
        self.characterize_mode = self._get_bool_param('characterize_mode')
        self.speed_factor = float(self.get_parameter('speed_factor').value)
        self.amplitude = float(self.get_parameter('amplitude').value)
        self.phase_offset = float(self.get_parameter('phase_offset').value)

        # Map joint -> actuator model
        self.joint_model = {}
        for i, model in enumerate(self.actuator_chain):
            if i < len(self.joint_names):
                self.joint_model[self.joint_names[i]] = model

        # Build per-joint gain profiles
        self.profiles = {}
        for joint in self.joint_names:
            model = self.joint_model.get(joint, 'x4_10')
            self.profiles[joint] = _merge_profile(model, joint)

        # ----------------------- timing / state -----------------------
        self.start_time = self.get_clock().now()
        self.last_print_time = self.get_clock().now()
        self.last_msg_time = None

        self.vel_alpha = 1.0
        self.current_cmd = {joint: 0.0 for joint in self.joint_names}
        self.latest_states = {joint: {'velocity': 0.0, 'effort': 0.0} for joint in self.joint_names}

        # Last-term breakdown for diagnostics
        self._last_target_v = {joint: 0.0 for joint in self.joint_names}
        self._last_ff = {joint: 0.0 for joint in self.joint_names}

        # ----------------------- publishers -----------------------
        self.cmd_pubs = []
        for topic in self.topic_names:
            self.cmd_pubs.append(self.create_publisher(Float64MultiArray, topic, 10))

        # ----------------------- subscriber -----------------------
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # ----------------------- characterization state machine -----
        self.char_joint_idx = 0
        self.char_phase = 'IDLE'
        self.char_torque = 0.0
        self.char_phase_start_time = self.get_clock().now()
        self.char_breakaway_count = 0
        self.char_phase_samples = []
        self.char_k_stiction_pos = None
        self.char_k_stiction_neg = None
        self.char_k_stiction = None
        self.char_tau_1 = None
        self.char_tau_2 = None
        self.char_v1 = None
        self.char_v2 = None
        self.char_results = {}
        self.char_results_printed = False

        mode_str = 'CHARACTERIZATION' if self.characterize_mode else 'pure feedforward wave'
        self.get_logger().info(
            'MultiActuatorTorqueController initialized (%s mode) | joints: %s | models: %s' %
            (mode_str, ', '.join(self.joint_names), ', '.join(self.actuator_chain) if self.actuator_chain else '(none)'))

    def _get_bool_param(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('true', '1', 'yes', 't')

    def joint_state_callback(self, msg: JointState):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            msg_time = now
        else:
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

        if self.last_msg_time is None:
            self.last_msg_time = msg_time

        dt = (msg_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = msg_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.01

        # Read states
        for i, name in enumerate(msg.name):
            if name in self.latest_states:
                vel = msg.velocity[i] if i < len(msg.velocity) else self.latest_states[name]['velocity']
                prev_vel = self.latest_states[name]['velocity']
                self.latest_states[name]['velocity'] = ((1.0 - self.vel_alpha) * prev_vel +
                                                        self.vel_alpha * vel)
                if i < len(msg.effort):
                    self.latest_states[name]['effort'] = msg.effort[i]

        cmd_array = [0.0] * len(self.joint_names)

        if self.characterize_mode:
            self._run_characterization(t, dt, now, cmd_array)
        else:
            self._run_pure_ff_control(t, cmd_array)

        cmd_msg = Float64MultiArray()
        cmd_msg.data = cmd_array
        for pub in self.cmd_pubs:
            pub.publish(cmd_msg)

        if (now - self.last_print_time).nanoseconds / 1e9 >= 1.0:
            self.last_print_time = now
            if self.characterize_mode:
                self._log_characterization_progress(t)
            else:
                self._log_diagnostics()

    # ------------------------------------------------------------------ #
    # Pure Feedforward Control Law (No P/I feedback fighting the motors)
    # ------------------------------------------------------------------ #
    def _run_pure_ff_control(self, t, cmd_array):
        ramp_scale = min(1.0, t / 2.0)

        for idx, joint in enumerate(self.joint_names):
            prof = self.profiles[joint]

            # Target velocity profile
            target_v = ramp_scale * self.amplitude * math.cos(
                self.speed_factor * t - idx * self.phase_offset)
            self._last_target_v[joint] = target_v

            # Coulomb stiction breakaway (with zero-crossing deadzone) + Viscous feedforward
            if abs(target_v) < 0.02:
                stiction_term = 0.0
            else:
                safe_stiction = min(prof['k_stiction'], prof['torque_limit'] * 0.4)
                stiction_term = safe_stiction * math.tanh(target_v * 15.0)

            viscous_term = prof['k_ff'] * target_v
            raw_cmd = stiction_term + viscous_term
            self._last_ff[joint] = raw_cmd

            # Clamp to safe per-joint torque limits
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

    # ------------------------------------------------------------------ #
    # Automated characterization (unchanged state machine)
    # ------------------------------------------------------------------ #
    def _run_characterization(self, t, dt, now, cmd_array):
        if self.char_phase == 'DONE' or self.char_joint_idx >= len(self.joint_names):
            self.char_phase = 'DONE'
            if not self.char_results_printed:
                self.char_results_printed = True
                self._print_characterization_results()
            return

        joint = self.joint_names[self.char_joint_idx]
        idx = self.char_joint_idx
        model = self.joint_model.get(joint, 'x4_10')
        prof = self.profiles[joint]
        tl = prof['torque_limit']

        elapsed_phase = (now - self.char_phase_start_time).nanoseconds / 1e9
        vel = self.latest_states[joint]['velocity']
        self.char_phase_samples.append((t, vel))

        if self.char_phase == 'IDLE':
            self.char_phase = 'PHASE1_POS'
            self.char_phase_start_time = now
            self.char_torque = 0.0
            self.char_breakaway_count = 0
            self.char_phase_samples = []
            return

        elif self.char_phase == 'PHASE1_POS':
            self.char_torque = min(tl, self.char_torque + 0.15 * dt)
            cmd_array[idx] = self.char_torque
            if abs(vel) > 0.08:
                self.char_breakaway_count += 1
            else:
                self.char_breakaway_count = 0
            if self.char_breakaway_count >= 3:
                self.char_k_stiction_pos = self.char_torque
                self._enter_phase1_pause(now)

        elif self.char_phase == 'PHASE1_PAUSE':
            cmd_array[idx] = 0.0
            if elapsed_phase >= 1.0:
                self.char_phase = 'PHASE1_NEG'
                self.char_phase_start_time = now
                self.char_torque = 0.0
                self.char_breakaway_count = 0
                self.char_phase_samples = []

        elif self.char_phase == 'PHASE1_NEG':
            self.char_torque = max(-tl, self.char_torque - 0.15 * dt)
            cmd_array[idx] = self.char_torque
            if abs(vel) > 0.08:
                self.char_breakaway_count += 1
            else:
                self.char_breakaway_count = 0
            if self.char_breakaway_count >= 3:
                self.char_k_stiction_neg = self.char_torque
                self.char_k_stiction = 0.85 * ((abs(self.char_k_stiction_pos) +
                                                abs(self.char_k_stiction_neg)) / 2.0)
                self.char_tau_1 = min(tl, self.char_k_stiction / 0.85 + 0.3)
                self.char_tau_2 = min(tl, self.char_k_stiction / 0.85 + 0.6)
                self.char_phase = 'PHASE2_TAU1'
                self.char_phase_start_time = now
                self.char_phase_samples = []

        elif self.char_phase in ('PHASE2_TAU1', 'PHASE2_TAU2'):
            tau = self.char_tau_1 if self.char_phase == 'PHASE2_TAU1' else self.char_tau_2
            cmd_array[idx] = tau
            if elapsed_phase >= 1.5:
                phase_start_t = (self.char_phase_start_time - self.start_time).nanoseconds / 1e9
                window_start_t = phase_start_t + 1.0
                recent = [v for (st, v) in self.char_phase_samples if st >= window_start_t]
                v_avg = sum(recent) / len(recent) if recent else 0.0

                if self.char_phase == 'PHASE2_TAU1':
                    self.char_v1 = v_avg
                    self.char_phase = 'PHASE2_TAU2'
                else:
                    self.char_v2 = v_avg
                    if abs(self.char_v2 - self.char_v1) > 0.05:
                        k_ff = max(0.01, (self.char_tau_2 - self.char_tau_1) /
                                 (self.char_v2 - self.char_v1))
                    else:
                        k_ff = prof['k_ff']
                    self.char_results[joint] = {
                        'k_stiction': self.char_k_stiction,
                        'k_ff': k_ff,
                    }
                    self.char_joint_idx += 1
                    self.char_phase = 'PHASE1_POS'
                    self.char_phase_start_time = now
                    self.char_torque = 0.0
                    self.char_breakaway_count = 0
                    self.char_k_stiction_pos = None
                    self.char_k_stiction_neg = None
                    self.char_k_stiction = None
                    self.char_tau_1 = None
                    self.char_tau_2 = None
                    self.char_v1 = None
                    self.char_v2 = None
                self.char_phase_samples = []

    def _enter_phase1_pause(self, now):
        self.char_phase = 'PHASE1_PAUSE'
        self.char_phase_start_time = now
        self.char_torque = 0.0
        self.char_breakaway_count = 0
        self.char_phase_samples = []

    def _log_characterization_progress(self, t):
        if self.char_phase == 'DONE':
            return
        joint = self.joint_names[self.char_joint_idx]
        vel = self.latest_states[joint]['velocity']
        line = (
            '[Characterize] joint=%s model=%s phase=%s t=%.2fs '
            'cmd_torque=%+6.3f N.m vel=%+6.3f rad/s'
        ) % (joint, self.joint_model.get(joint, '?'), self.char_phase, t,
             self.char_torque, vel)
        if self.char_k_stiction is not None:
            line += ' k_stiction=%.4f' % self.char_k_stiction
        self.get_logger().info(line)

    def _print_characterization_results(self):
        lines = []
        lines.append('\n' + '=' * 60)
        lines.append('AUTOMATED CHARACTERIZATION COMPLETE')
        lines.append('=' * 60)
        lines.append('Copy-paste the block below into JOINT_CHARACTERIZATION_OVERRIDES:')
        lines.append('JOINT_CHARACTERIZATION_OVERRIDES = {')
        for joint in self.joint_names:
            r = self.char_results.get(joint, {'k_stiction': 0.0, 'k_ff': 0.0})
            lines.append("    '%s': {'k_stiction': %.4f, 'k_ff': %.4f}," %
                         (joint, r['k_stiction'], r['k_ff']))
        lines.append('}')
        lines.append('All joints commanded to 0.0 N.m.')
        lines.append('=' * 60 + '\n')
        self.get_logger().info("\n".join(lines))

    def shutdown(self):
        zero_msg = Float64MultiArray(data=[0.0] * len(self.joint_names))
        for pub in self.cmd_pubs:
            pub.publish(zero_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MultiActuatorTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.shutdown()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()