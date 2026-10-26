#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class MultiActuatorTorqueController(Node):
    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        self.declare_parameter('joint_names', ['shoulder_joint', 'elbow_joint', 'wrist_joint'])
        self.declare_parameter('topic_names', ['/effort_controller/commands'])
        self.declare_parameter('torque_limit', 1.5)
        
        self.joint_names = self.get_parameter('joint_names').get_parameter_value().string_array_value
        self.topic_names = self.get_parameter('topic_names').get_parameter_value().string_array_value
        self.torque_limit = self.get_parameter('torque_limit').get_parameter_value().double_value

        self.start_time = self.get_clock().now()
        
        self.k_ff = {
            'shoulder_joint': 1, 
            'elbow_joint': 0.9,    
            'wrist_joint': 1.3     
        }
        self.kp = 0.12
        self.ki = 0.12   
        self.max_i_term = 0.177
        
        self.vel_alpha = 1.0
        
        self.integral_error = {joint: 0.0 for joint in self.joint_names}
        self.current_cmd = {joint: 0.0 for joint in self.joint_names}
        
        self.last_print_time = self.get_clock().now()
        self.last_msg_time = None

        self.cmd_pubs = []
        for topic in self.topic_names:
            self.cmd_pubs.append(self.create_publisher(Float64MultiArray, topic, 10))

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.latest_states = {joint: {'velocity': 0.0, 'effort': 0.0} for joint in self.joint_names}

    def joint_state_callback(self, msg: JointState):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        if self.last_msg_time is None:
            self.last_msg_time = msg_time
            return
            
        dt = (msg_time - self.last_msg_time).nanoseconds / 1e9
        self.last_msg_time = msg_time

        if dt <= 0.0 or dt > 0.5:
            return

        # 1. Read states
        for i, name in enumerate(msg.name):
            if name in self.latest_states:
                vel = msg.velocity[i]
                prev_vel = self.latest_states[name]['velocity']
                self.latest_states[name]['velocity'] = (1.0 - self.vel_alpha) * prev_vel + (self.vel_alpha * vel)
                if i < len(msg.effort):
                    self.latest_states[name]['effort'] = msg.effort[i]

        cmd_msg = Float64MultiArray()
        cmd_array = []
        
        # Phase-shifted target velocities for a biological wave motion
        # INCREASE speed_factor for faster oscillation
        # DECREASE amplitude for less range of motion
        speed_factor = 3
        amplitude = 1
        
        target_v_shared = amplitude * math.cos(speed_factor * t)
        target_vels = {joint: target_v_shared for joint in self.joint_names}

        # 2. Compute Control Law (FF + P + Clamped I)
        for joint in self.joint_names:
            current_vel = self.latest_states[joint]['velocity']
            target_v = target_vels[joint]
            error = target_v - current_vel
            
            # Smooth Feedforward scaling avoids violent torque snaps at zero-crossings
            ff_term = self.k_ff[joint] * target_v
            
            # Proportional term
            p_term = self.kp * error
            
            # Integrate and tightly clamp the I-term directly to prevent stiction windup
            self.integral_error[joint] += error * dt
            max_err = self.max_i_term / self.ki if self.ki > 0.0 else 0.0
            self.integral_error[joint] = max(-max_err, min(max_err, self.integral_error[joint]))
            i_term = self.ki * self.integral_error[joint]
            
            raw_cmd = ff_term + p_term + i_term
            clamped_cmd = max(-self.torque_limit, min(self.torque_limit, raw_cmd))

            self.current_cmd[joint] = clamped_cmd
            cmd_array.append(clamped_cmd)

        # 3. Publish Immediately
        cmd_msg.data = cmd_array
        for pub in self.cmd_pubs:
            pub.publish(cmd_msg)

        # 4. Diagnostic Logging
        if (now - self.last_print_time).nanoseconds / 1e9 >= 1.0:
            self.last_print_time = now
            log_lines = ["\n"]
            for joint in self.joint_names:
                vel = self.latest_states[joint]['velocity']
                eff = self.latest_states[joint]['effort']
                cmd = self.current_cmd[joint]
                target = target_vels[joint]
                log_lines.append(
                    f"[{joint:14s}]Tgt Vel: {target:+.2f}, Act Vel: {vel:+.2f}, Cmd Eff: {cmd:+.3f} N·m"
                )
            self.get_logger().info("\n".join(log_lines))

def main(args=None):
    rclpy.init(args=args)
    node = MultiActuatorTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_msg = Float64MultiArray(data=[0.0] * len(node.joint_names))
        for pub in node.cmd_pubs:
            pub.publish(zero_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()