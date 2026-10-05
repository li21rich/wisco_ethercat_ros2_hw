# WiscoHumanoids Hardware Interface

This repository contains the physical hardware configuration, PDO mapping, and ROS 2 launch files to bridge the RK3588's real-time EtherCAT master to the Jetson Thor's high-level AI policy.

## 🏗️ System Architecture

The software is split into hardware definitions (this repo) and mathematical control plugins (loaded dynamically).

```text
wisco_ws/src/
├── ethercat_driver_ros2/          # (add separately)
│
├── wisco_hw_interface/            # (THIS REPO: Hardware configuration)
│   ├── config/rmd_x4_10.yaml      # Hexadecimal PDO maps & gear ratios
│   ├── config/controllers.yaml    # Dictates WHICH controllers to load
│   ├── launch/test_bench.launch.py 
│   ├── urdf/test_bench.urdf.xacro # Physical robot description & slave IDs
│   └── README.md
│
└── wisco_controllers/             # (controls stack to be added separately)
    ├── include/wisco_controllers/
    │   └── wbc_controller.hpp     # Class headers, Eigen matrix declarations
    └── src/
        └── wbc_controller.cpp     # The 1kHz QP solver & Pinocchio rigid body dynamics

 Quick Start Workflow
1. Build the Workspace

Run this from the workspace root (~/Code/wisco_ws) whenever you add new files, modify CMakeLists.txt, or compile C++ code.
The --symlink-install flag ensures Python and YAML edits update automatically without requiring a rebuild.
Bash

colcon build --symlink-install

2. Source the Environment

Run this in every new terminal before executing ROS 2 commands.
Bash

source /opt/ros/jazzy/setup.bash
source ~/Code/wisco_ws/install/setup.bash

3. Launch the Stack

Requires sudo to allow the EtherCAT driver to access the raw /dev/EtherCAT0 network device.
Bash

# 1. Start the kernel module (ensures the USB-C dongle is bound)
sudo /etc/init.d/ethercat start

# 2. Launch ros2_control and the hardware interface
sudo -E bash -c 'source /opt/ros/jazzy/setup.bash && source ~/Code/wisco_ws/install/setup.bash && ros2 launch wisco_hw_interface test_bench.launch.py'

Useful Debugging Commands

Open a second (sourced) terminal while the launch file is running to inspect the real-time loop:

    View active network data: ros2 topic list

    Stream live motor telemetry (pos/vel/effort): ros2 topic echo /joint_states

    Verify hardware loaded: ros2 control list_hardware_interfaces

    Verify controllers loaded: ros2 control list_controllers

    Command torque manually: ros2 topic pub /effort_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.5]}"
