import os
import tempfile
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnShutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

# ========================================
# Actuation priority order (must match xacro actuated_joint_names).
# First N joints receive hardware actuators, where N = len(ACTUATOR_CHAIN).
# Order: both arms (14), both legs (12), torso (3), head (2) = 31 joints.
# ========================================
ACTUATED_JOINT_NAMES = [
    # Left arm
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
    'left_elbow', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_roll',
    # Right arm
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_roll',
    # Left leg
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
    'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    # Right leg
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
    'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    # Torso
    'torso_yaw', 'torso_roll', 'torso_pitch',
    # Head
    'head_yaw', 'head_pitch',
]

# ========================================
# Ordered list of actuator models in series. To scale the hardware chain,
# edit ONLY this list. joint_names are derived automatically from the
# actuation priority order above.
# ========================================
ACTUATOR_CHAIN = ['x4_10', 'x4_10', 'x4_36', 'x4_36']

# First N joints from actuation order get hardware actuators.
JOINT_NAMES = ACTUATED_JOINT_NAMES[:len(ACTUATOR_CHAIN)]

# Model counts derived from ACTUATOR_CHAIN for the xacro description.
N_X4_10 = ACTUATOR_CHAIN.count('x4_10')
N_X4_36 = ACTUATOR_CHAIN.count('x4_36')
N_X6_60 = ACTUATOR_CHAIN.count('x6_60')
N_X8_120 = ACTUATOR_CHAIN.count('x8_120')


def _make_temp_file_cleanup(temp_yaml_path):
    """Return a launch action that deletes the dynamically generated temp YAML on shutdown."""

    def cleanup(context):
        try:
            os.unlink(temp_yaml_path)
        except OSError:
            pass
        return []

    return OpaqueFunction(function=cleanup)


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
    # A merged copy of controllers.yaml is written to a temp file with the
    # joints injected under controller_manager.ros__parameters.effort_controller
    # so the parameter namespaces match what controller_manager expects.
    # ------------------------------------------------------------------
    with open(base_controllers_yaml, 'r') as f:
        controllers_config = yaml.safe_load(f)
    controllers_config = controllers_config or {}
    ec = controllers_config.setdefault('effort_controller', {}).setdefault('ros__parameters', {})
    ec['joints'] = JOINT_NAMES
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.safe_dump(controllers_config, temp_yaml)
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
        RegisterEventHandler(
            OnShutdown(on_shutdown=[_make_temp_file_cleanup(temp_yaml_path)])
        ),

        # ros2_control_node: hardware interface + controller configuration.
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                temp_yaml_path,
                {'robot_description': robot_desc},
            ],
            output='both',
        ),

        # robot_state_publisher.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),

        # Spawn controllers
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['effort_controller', '--controller-manager', '/controller_manager'],
        ),

        # Multi-actuator torque command node.
        Node(
            package='wisco_ethercat_ros2_hw',
            executable='multiple_act_command.py',
            name='multiple_act_command_name',
            output='screen',
            output_format='{line}',
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
