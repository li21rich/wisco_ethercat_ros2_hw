cmake_minimum_required(VERSION 3.8)
project(wisco_ethercat_ros2_hw)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# --- example_ros_node: minimal ROS API that awaits torque commands ---
add_executable(example_ros_node src/example_ros_node.cpp)
target_compile_features(example_ros_node PRIVATE cxx_std_17)
ament_target_dependencies(example_ros_node rclcpp std_msgs)
install(TARGETS example_ros_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
install(
  DIRECTORY config urdf launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/example_torque_command_ros_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME example_torque_command_ros_node
)

install(
  PROGRAMS scripts/multiple_act_command.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/multi_velocity_cmd.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
