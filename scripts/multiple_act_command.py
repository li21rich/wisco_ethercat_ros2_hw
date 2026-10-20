#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

limit = 5.0  # Example torque limit for each actuator in N·m

class MultiActuatorTorqueController(Node):
    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        self.declare_parameter('joint_names', ['shoulder_joint', 'elbow_joint', 'wrist_joint'])
        self.declare_parameter('topic_names', ['/effort_controller/commands'])
        self.declare_parameter('torque_limit', limit)
        self.joint_names = self.get_parameter('joint_names').get_parameter_value().string_array_value
        self.topic_names = self.get_parameter('topic_names').get_parameter_value().string_array_value
        self.torque_limit = self.get_parameter('torque_limit').get_parameter_value().double_value

        self.cmd_pubs = []
        for topic in self.topic_names:
            pub = self.create_publisher(Float64MultiArray, topic, 10)
            self.cmd_pubs.append(pub)

        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )

        self.latest_states = {joint: {'position': 0.0, 'velocity': 0.0, 'effort': 0.0} for joint in self.joint_names}
        self.timer = self.create_timer(0.01, self.control_loop_callback) # 100Hz loop

        self.get_logger().info(f'Initialized Multi-Actuator Controller for joints: {self.joint_names}')

    def joint_state_callback(self, msg: JointState):
        for name, pos, vel, eff in zip(msg.name, msg.position, msg.velocity, msg.effort):
                    if name in self.latest_states:
                        self.latest_states[name] = {'position': pos, 'velocity': vel, 'effort': eff}

    def control_loop_callback(self):
        cmd_msg = Float64MultiArray()
        # Create a list of command values for all joints
        cmd_msg.data = [0.0] * len(self.joint_names)
        
        for i, joint in enumerate(self.joint_names):
            if abs(cmd_msg.data[i]) > self.torque_limit:
                self.get_logger().warn(
                    f'Torque command for {joint} exceeds limit. Capping to {self.torque_limit} N·m.'
                )
                cmd_msg.data[i] = self.torque_limit if cmd_msg.data[i] > 0 else -self.torque_limit
        
        # Send the command list to all joints
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