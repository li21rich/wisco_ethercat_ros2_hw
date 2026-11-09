import os
import tempfile
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

# ========================================
# Ordered list of actuator models in series.  To scale the hardware chain,
# edit ONLY this list (and per-joint overrides in the controller script).
# joint_names and the xacro model counts are derived automatically below.
ACTUATOR_CHAIN = ['x4_10', 'x4_10', 'x4_36', 'x4_36']

JOINT_NAMES = [f'joint_{i + 1}' for i in range(len(ACTUATOR_CHAIN))]

# Model counts derived from ACTUATOR_CHAIN for the xacro description.
N_X4_10 = ACTUATOR_CHAIN.count('x4_10')
N_X4_36 = ACTUATOR_CHAIN.count('x4_36')
N_X6_60 = ACTUATOR_CHAIN.count('x6_60')
N_X8_120 = ACTUATOR_CHAIN.count('x8_120')
# ========================================


def generate_launch_description():
    pkg_share = get_package_share_directory('wisco_ethercat_ros2_hw')
    urdf_file = os.path.join(pkg_share, 'urdf', 'test_bench.urdf.xacro')
    base_controllers_yaml = os.path.join(pkg_share, 'config', 'controllers.yaml')

    # Robot description built from the derived model counts.
    robot_desc = ParameterValue(
        Command([
            'xacro ', urdf_file,
            f' n_x4_10:={N_X4_10}',
            f' n_x4_36:={N_X4_36}',
            f' n_x6_60:={N_X6_60}',
            f' n_x8_120:={N_X8_120}',
        ]),
        value_type=str,
    )

    # ------------------------------------------------------------------
    # Dynamic effort_controller joint-list injection.
    #
    # The joints parameter is written at launch time from ACTUATOR_CHAIN so
    # that config/controllers.yaml never needs to be edited by hand again.
    # The temp file is passed to BOTH ros2_control_node and the effort_controller
    # spawner (via --param-file) so the joint list stays consistent everywhere.
    # ------------------------------------------------------------------
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.safe_dump({
        'effort_controller': {
            'ros__parameters': {
                'joints': JOINT_NAMES,
            }
        }
    }, temp_yaml)
    temp_yaml.close()
    temp_yaml_path = temp_yaml.name

    # LaunchArgument: run automated per-joint characterization from the CLI.
    characterize_mode_arg = DeclareLaunchArgument(
        'characterize_mode',
        default_value='false',
        description='Enable automated per-joint stiction/viscous characterization',
    )

    return LaunchDescription([
        characterize_mode_arg,

        # ros2_control_node: hardware interface + controller configuration.
        # base_controllers_yaml provides the controller_manager + controller type
        # definitions; temp_yaml supplies the (dynamically generated) joints list.
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[base_controllers_yaml, temp_yaml_path],
            output='both',
        ),

        # robot_state_publisher.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),

        # Spawn the standard broadcasters / controllers.
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'effort_controller',
                '--controller-manager', '/controller_manager',
                '--param-file', temp_yaml_path,
            ],
        ),

        # Multi-actuator torque command node.
        Node(
            package='wisco_ethercat_ros2_hw',
            executable='multiple_act_command_executable',
            name='multiple_act_command_name',
            output='screen',
            parameters=[
                {'joint_names': JOINT_NAMES},
                {'actuator_chain': ACTUATOR_CHAIN},
                {'characterize_mode': LaunchConfiguration('characterize_mode')},
                {'speed_factor': 1.5},
                {'amplitude': 1.0},
                {'phase_offset': 0.4},
            ],
        ),
    ])
