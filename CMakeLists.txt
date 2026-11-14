cmake_minimum_required(VERSION 3.8)
project(wisco_ethercat_ros2_hw)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)

# --- SafeCommandController: the command controller the launch file loads ---
add_library(wisco_ethercat_ros2_hw_controllers SHARED src/safe_command_controller.cpp)
target_compile_features(wisco_ethercat_ros2_hw_controllers PUBLIC cxx_std_17)
target_include_directories(wisco_ethercat_ros2_hw_controllers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(wisco_ethercat_ros2_hw_controllers
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle realtime_tools
  std_msgs)
pluginlib_export_plugin_description_file(controller_interface safe_command_controller_plugin.xml)
install(TARGETS wisco_ethercat_ros2_hw_controllers
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

# --- example_ros_node: minimal ROS API that awaits torque commands ---
add_executable(example_ros_node src/example_ros_node.cpp)
target_compile_features(example_ros_node PRIVATE cxx_std_17)
ament_target_dependencies(example_ros_node rclcpp std_msgs)
install(TARGETS example_ros_node DESTINATION lib/${PROJECT_NAME})

# --- Python: config generation used by the launch file and scripts ---
ament_python_install_package(${PROJECT_NAME})

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
  PROGRAMS
    scripts/multiple_act_command.py
    scripts/multi_velocity_cmd.py
    scripts/bringup_check.py
    scripts/check_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_safety_logic test/test_safety_logic.cpp)
  target_include_directories(test_safety_logic PRIVATE include)

  ament_add_pytest_test(test_hardware_config test/test_hardware_config.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(controller_interface hardware_interface pluginlib rclcpp
  rclcpp_lifecycle realtime_tools std_msgs)
ament_package()
