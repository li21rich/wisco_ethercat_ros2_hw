// ArmController: the arm stack's RT loop (spec Mode P / Mode T) as a
// ros2_control controller. Runs arm_law.hpp every cycle with Pinocchio
// gravity from the robot description plus the calibration file.
//
// Topics (Float64MultiArray; layouts are mirrored in arm_stack/hw/ros_client.py):
//   ~/command (in)   [mode (0 P, 1 T), flags (1 gravity_ff, 2 ignore_soft_limits),
//                     friction_ff_gain, q[n], qd[n], kp[n], kd[n], tau_ff[n],
//                     torque_clamp[n]]
//                    NaN = not set. A command older than watchdog_s (50 ms)
//                    drops the arm into gravity comp + damping in CST.
//   ~/state (out)    [t (s, steady), mode, fallback, limit_hit, clamped,
//                     q[n], qd[n], tau_measured[n], tau_out[n], q_target[n]]
//                    every cycle, through a realtime publisher (try-lock; the
//                    non-RT thread publishes, the loop never blocks).
//
// Interfaces per joint: command position, effort, torque_offset,
// mode_of_operation; state position, velocity, effort.
#ifndef WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "wisco_ethercat_ros2_hw/arm_law.hpp"

namespace wisco_ethercat_ros2_hw
{

class GravityModel;

constexpr size_t kCommandHeader = 3;
constexpr size_t kCommandBlocks = 6;
constexpr size_t kStateHeader = 5;
constexpr size_t kStateBlocks = 5;

class ArmController : public controller_interface::ControllerInterface
{
public:
  ArmController();
  ~ArmController() override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct Command
  {
    arm::LawCommand law;
    int64_t received_ns = 0;
  };
  using Ref = std::reference_wrapper<hardware_interface::LoanedCommandInterface>;
  using StateRef = std::reference_wrapper<hardware_interface::LoanedStateInterface>;

  static int64_t steady_now_ns();

  std::vector<std::string> joints_;
  arm::LawConfig cfg_;
  std::unique_ptr<GravityModel> gravity_;
  double blend_s_ = 0.02;

  realtime_tools::RealtimeBuffer<std::shared_ptr<Command>> command_buffer_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> state_pub_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> rt_state_;

  std::vector<Ref> cmd_pos_, cmd_eff_, cmd_off_, cmd_mode_;
  std::vector<StateRef> st_pos_, st_vel_, st_eff_;

  // preallocated per-cycle buffers
  std::vector<double> q_, qd_, tau_meas_, g_, tau_switch_, q_target_prev_;
  arm::LawOutput out_;
  int last_mode_ = -1;
  int64_t switch_ns_ = 0;
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_
