#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class MultiActuatorTorqueController(Node):
    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        # TODO(Multi-Actuator): Make target joints and command topics configurable via ROS2 parameters
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3'], 
                               'topic_names', ['/effort_controller/commands'])
        self.joint_names = self.get_parameter('joint_names').get_parameter_value().string_array_value
        self.topic_names = self.get_parameter('topic_names').get_parameter_value().string_array_value

        # TODO(Multi-Actuator): Update publisher to send dynamic size effort array matching number of daisy-chained actuators
        self.cmd_pubs = []
        for topic in self.topic_names:
            pub = self.create_publisher(Float64MultiArray, topic, 10)
            self.cmd_pubs.append(pub)

        # TODO(Multi-Actuator): Subscribe to /joint_states and parse position/velocity/effort telemetry across all joints
        for joint in self.joint_names:
            self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # TODO(Multi-Actuator): Adapt control loop timer and safety monitoring routines for concurrent multi-joint execution
        self.timer = self.create_timer(0.01, self.control_loop_callback) # 100Hz loop

        self.get_logger().info(f'Initialized Multi-Actuator Controller for joints: {self.joint_names}')

    def joint_state_callback(self, msg: JointState):
        # TODO(Multi-Actuator): Extract multi-element arrays for position, velocity, and effort based on active joint ordering
        self.get_logger().debug(f'Received joint states: {msg.name} | Positions: {msg.position} | Velocities: {msg.velocity} | Efforts: {msg.effort}')

    def control_loop_callback(self):
        # TODO(Multi-Actuator): Calculate torque vector commands (e.g. stiction breaking, torque trajectory) for 2 to 3 drives
        cmd_msg = Float64MultiArray()
        for joint in self.joint_names:
            # Placeholder logic for torque command; replace with actual control algorithm
            torque_command = 0.0  # Replace with computed torque for each joint
            cmd_msg.data.append(torque_command)

        # Example dummy array for 3 joints
        cmd_msg.data = [0.0] * len(self.joint_names)
        
        # TODO(Multi-Actuator): Implement safety checks (e.g., effort limiters, velocity caps) across all actuators before publishing
        for i, joint in enumerate(self.joint_names):
            if abs(cmd_msg.data[i]) > 5.0:  # Example effort limit
                self.get_logger().warn(f'Torque command for {joint} exceeds limit. Capping to 5.0 N·m.')
                cmd_msg.data[i] = 5.0 if cmd_msg.data[i] > 0 else -5.0
        for pub in self.cmd_pubs:
            pub.publish(cmd_msg)

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