#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class MultiActuatorTorqueController(Node):
    def __init__(self):
        super().__init__('multi_actuator_torque_controller')

        # TODO(Multi-Actuator): Make target joints and command topics configurable via ROS2 parameters
        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3'])
        self.joint_names = self.get_parameter('joint_names').get_parameter_value().string_array_value

        # TODO(Multi-Actuator): Update publisher to send dynamic size effort array matching number of daisy-chained actuators
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/effort_controller/commands',
            10
        )

        # TODO(Multi-Actuator): Subscribe to /joint_states and parse position/velocity/effort telemetry across all joints
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # TODO(Multi-Actuator): Adapt control loop timer and safety monitoring routines for concurrent multi-joint execution
        self.timer = self.create_timer(0.01, self.control_loop_callback) # 100Hz loop

        self.get_logger().info(f'Initialized Multi-Actuator Controller for joints: {self.joint_names}')

    def joint_state_callback(self, msg: JointState):
        # TODO(Multi-Actuator): Extract multi-element arrays for position, velocity, and effort based on active joint ordering
        pass

    def control_loop_callback(self):
        # TODO(Multi-Actuator): Calculate torque vector commands (e.g. stiction breaking, torque trajectory) for 2 to 3 drives
        cmd_msg = Float64MultiArray()
        # Example dummy array for 3 joints
        cmd_msg.data = [0.0] * len(self.joint_names)
        
        # TODO(Multi-Actuator): Implement safety checks (e.g., effort limiters, velocity caps) across all actuators before publishing
        self.cmd_pub.publish(cmd_msg)

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