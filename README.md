# WiscoHumanoids Hardware Interface

Demos: 
https://drive.google.com/file/d/1t2eEQImacynxiRTeHf_oAAU96mAUvgAz/view
https://drive.google.com/file/d/1zwNxs48vFlNf9jD_jDFN_XyyGplYSpis/view

This repository contains the physical hardware configuration, PDO mapping, demo scripts, and ROS 2 launch files required to bridge the RK3588's real-time EtherCAT master to the Jetson Thor's high-level AI policy.

Currently, it serves as a test bench demonstrating single- or multi-servo control (specifically MyActuator RMD X-Series motors) using ROS 2 Jazzy and ethercat_driver_ros2 on a standard Linux laptop (which will be replaced by the RK3588 in the final implementation). Because this demo was developed on a machine without a PREEMPT_RT kernel patch, it is configured to run at 100Hz rather than the intended 1kHz.

## System Architecture

RK3588 (Real-Time Control & Hardware Interface)

On the RK3588, the software stack is split into static hardware definitions (this repository) and dynamically loaded mathematical control plugins:

```text
wisco_ws/src/
├── ethercat_driver_ros2/          # (add separately, you need to install this for it to work github.com/ICube-Robotics/ethercat_driver_ros2)
│
├── wisco_ethercat_ros2_hw/        # <----- (THIS REPO: Hardware configuration)
│   ├── config/rmd_x4_10.yaml      #      Hexadecimal PDO maps & gear ratios
│   ├── config/rmd_x4_36.yaml      #      Hexadecimal PDO maps & gear ratios
│   ├── config/rmd_x6_60.yaml      #      Hexadecimal PDO maps & gear ratios
│   ├── config/rmd_x8_120.yaml     #      Hexadecimal PDO maps & gear ratios
│   ├── config/controllers.yaml    #      Dictates which controllers to load
│   ├── launch/test_bench.launch.py 
│   ├── urdf/test_bench.urdf.xacro #      Physical robot description & slave IDs
│   └── README.md                  # (you are here)
│
└── wisco_controllers/             # (C++ controls stack to be added separately later along with the rest of the humanoid)
    ├── include/wisco_controllers/
    │   └── wbc_controller.hpp     # Class headers, Eigen matrix declarations
    └── src/
        └── wbc_controller.cpp     # The 1kHz QP solver, Pinocchio rigid body dynamics, etc.

ros2_control (controller_manager process, same process as hw_interface)
├── ethercat_driver_ros2 (SystemInterface) ── 14x RMD-X4-10 CiA 402
├── wbc_controller plugin
└── State/Command Interfaces (hardware interface exposed by ethercat_driver_ros2, consumed/written by wbc_controller)
```
Runtime Process Structure on RK3588:
```text
ROS 2 (minimal) on RK3588
└── Command surface exposed by wbc_controller (action/topic for
    Cartesian/task targets) — receives targets from Jetson Thor
```

Jetson Thor (High-Level AI. Shown here for context):
```text
jetson_ws/src/
└── wisco_policy/
    ├── wisco_policy/
    │   └── policy_node.py              # RL locomotion policy / MPC
    └── launch/
        └── policy.launch.py

ROS 2 (full) on Jetson Thor
├── MoveIt 2 / AI pipeline
├── RL locomotion policy / MPC
├── Vision / SLAM
└── Sends high-level targets (50–100Hz) over DDS ──┐
                                                     │
                                                     v
                               ROS 2 DDS Network (non-RT link)
                                                     │
                                                     v
                     RK3588: wbc_controller's ROS 2 command surface
```

## Quick Start 

Note: In the commands below, ~/Code is used as a placeholder for the parent directory containing your wisco_ws workspace.

```bash
# Source the ROS 2 environment and build from the workspace root (e.g., ~/Code/wisco_ws):
source /opt/ros/jazzy/setup.bash

colcon build --symlink-install && source ~/Code/wisco_ws/install/setup.bash 
# ^ Build After any changes you make

""" 
Starting will require sudo so the EtherCAT driver can access the raw /dev/EtherCAT0 network device. To run without sudo use: 
$ echo 'KERNEL=="EtherCAT[0-9]*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ethercat.rules
$ sudo udevadm control --reload-rules && sudo udevadm trigger
It is also recommended to configure limits.conf
"""

# Starting the EtherCAT kernel module (make sure the slave is connected)
systemctl start ethercat
# Launch ros2_control and hardware interface
ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py
```

By default, test_bench.launch.py runs an example Python script that sends periodic torque commands to spin the motor every few seconds.

rviz2 is recommended to visualize the servo state in real-time.

You can also run a script manually in a separate terminal:
```bash
ros2 run wisco_ethercat_ros2_hw multiple_act_command_executable
```



## Demo Scripts

The package includes helper scripts to test actuator response without launching the full stack:

- **`multiple_act_command.py`**: Tester for many motors (no limit)
- **`example_torque_command_ros_node.py`**: Tester for one motor (deprecated)

## To add/update actuators in use:
1. test_bench.launch.py: update the actuator list
2. controllers.yaml: update the joint list
3. launch multiple_act_command and adjust override params

## Useful Debugging Commands

Open a second (sourced) terminal while the launch file is running to inspect the real-time loop:

* List active ROS2 topics: `ros2 topic list` (some of these you must be sudo)
* Stream live motor telemetry (pos/vel/effort): `ros2 topic echo /joint_states`
* Command torque manually (for example_torque_command_ros_node.py only): `ros2 topic pub /effort_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.5]}"`
* Check domains: `ethercat domains` (i.e. for 3 actuators it should say "Domain0: LogBaseAddr 0x00000000, Size  96, WorkingCounter 9/9")
* Latency analysis: `ros2 topic hz /joint_states` (publish rate, min/max latency, and standard deviation)
