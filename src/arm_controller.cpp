#include "wisco_ethercat_ros2_hw/arm_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>

#include "controller_interface/helpers.hpp"
#include "wisco_ethercat_ros2_hw/gravity_model.hpp"

namespace wisco_ethercat_ros2_hw
{

ArmController::ArmController() = default;
ArmController::~ArmController() = default;

int64_t ArmController::steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

controller_interface::CallbackReturn ArmController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>{});
    auto_declare<std::vector<double>>("max_torque", std::vector<double>{});
    auto_declare<std::vector<double>>("max_velocity", std::vector<double>{});
    auto_declare<std::vector<double>>("soft_lower", std::vector<double>{});
    auto_declare<std::vector<double>>("soft_upper", std::vector<double>{});
    auto_declare<std::vector<double>>("kp", std::vector<double>{});
    auto_declare<std::vector<double>>("kd", std::vector<double>{});
    auto_declare<std::vector<double>>("kd_fallback", std::vector<double>{});
    auto_declare<double>("watchdog_s", 0.05);
    auto_declare<double>("soft_margin", 0.05);
    auto_declare<double>("blend_s", 0.02);
    auto_declare<bool>("drive_feedback", false);
    auto_declare<bool>("hold_on_startup", false);
    auto_declare<bool>("hold_on_timeout", false);
    auto_declare<bool>("torque_compensation", true);
    auto_declare<bool>("gravity_enabled", true);
    auto_declare<bool>("hold_only", false);
    auto_declare<bool>("position_only", false);
    for (const char * name : {"local_center", "local_bias", "local_jacobian", "local_lower", "local_upper"}) {
      auto_declare<std::vector<double>>(name, std::vector<double>{});
    }
    auto_declare<bool>("write_max_torque", false);
    auto_declare<std::vector<bool>>("torque_offset_enabled", std::vector<bool>{});
    auto_declare<double>("velocity_ff_gain", 1.0);
    auto_declare<double>("velocity_ff_tau_s", 0.02);
    auto_declare<std::vector<double>>("coulomb_ff_nm", std::vector<double>{});
    auto_declare<std::string>("calibration_file", "");
    auto_declare<std::string>("robot_description", "");
    auto_declare<std::vector<std::string>>("link_mass_names", std::vector<std::string>{});
    auto_declare<std::vector<double>>("link_mass_values", std::vector<double>{});
    auto_declare<std::string>("payload_frame", "tool0");
    auto_declare<std::vector<std::string>>("keep_out_joint", std::vector<std::string>{});
    auto_declare<std::vector<std::string>>("keep_out_gate", std::vector<std::string>{});
    for (const char * name : {"keep_out_lower", "keep_out_upper", "keep_out_gate_min"}) {
      auto_declare<std::vector<double>>(name, std::vector<double>{});
    }
  } catch (const std::exception & e) {
    std::fprintf(stderr, "ArmController: exception in on_init: %s\n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ArmController::on_configure(const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  auto logger = node->get_logger();
  joints_ = node->get_parameter("joints").as_string_array();
  const size_t n = joints_.size();
  if (n == 0) {
    RCLCPP_ERROR(logger, "'joints' is empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  auto vec = [&](const char * name, std::vector<double> & out) {
      out = node->get_parameter(name).as_double_array();
      if (out.size() != n) {
        RCLCPP_ERROR(logger, "'%s' needs %zu entries, got %zu", name, n, out.size());
        return false;
      }
      return true;
    };
  std::vector<double> kp, kd;
  if (!vec("max_torque", cfg_.max_torque) || !vec("max_velocity", cfg_.max_velocity) ||
    !vec("soft_lower", cfg_.lower) || !vec("soft_upper", cfg_.upper) ||
    !vec("kd_fallback", cfg_.kd_fallback))
  {
    return controller_interface::CallbackReturn::ERROR;
  }
  kp_cfg_ = node->get_parameter("kp").as_double_array();
  kd_cfg_ = node->get_parameter("kd").as_double_array();
  if (kp_cfg_.size() != n) {kp_cfg_.assign(n, 0.0);}
  if (kd_cfg_.size() != n) {kd_cfg_ = cfg_.kd_fallback;}
  for (size_t i = 0; i < n; ++i) {
    if (!(cfg_.max_torque[i] > 0) || !(cfg_.max_velocity[i] > 0) || !(cfg_.lower[i] < cfg_.upper[i])) {
      RCLCPP_ERROR(logger, "bad limits for joint '%s'", joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  {
    const auto kj = node->get_parameter("keep_out_joint").as_string_array();
    const auto kg = node->get_parameter("keep_out_gate").as_string_array();
    const auto klo = node->get_parameter("keep_out_lower").as_double_array();
    const auto khi = node->get_parameter("keep_out_upper").as_double_array();
    const auto kmin = node->get_parameter("keep_out_gate_min").as_double_array();
    const size_t k = kj.size();
    if (kg.size() != k || klo.size() != k || khi.size() != k || kmin.size() != k) {
      RCLCPP_ERROR(logger, "keep_out_* parameters need the same length");
      return controller_interface::CallbackReturn::ERROR;
    }
    cfg_.keep_out.clear();
    auto index = [&](const std::string & name) {
        return static_cast<size_t>(std::find(joints_.begin(), joints_.end(), name) - joints_.begin());
      };
    for (size_t z = 0; z < k; ++z) {
      arm::KeepOut ko;
      ko.joint = index(kj[z]);
      ko.gate = index(kg[z]);
      ko.lower = klo[z];
      ko.upper = khi[z];
      ko.gate_min = kmin[z];
      if (ko.joint >= n || ko.gate >= n || ko.joint == ko.gate || !(ko.lower < ko.upper) ||
        !std::isfinite(ko.gate_min))
      {
        RCLCPP_ERROR(logger, "bad keep-out zone %zu (%s gated by %s)", z, kj[z].c_str(), kg[z].c_str());
        return controller_interface::CallbackReturn::ERROR;
      }
      cfg_.keep_out.push_back(ko);
      RCLCPP_INFO(
        logger, "keep-out: %s within [%.1f, %.1f] deg while %s < %.1f deg", kj[z].c_str(),
        ko.lower * 180 / M_PI, ko.upper * 180 / M_PI, kg[z].c_str(), ko.gate_min * 180 / M_PI);
    }
  }
  cfg_.coulomb.assign(n, 0.0);
  cfg_.viscous.assign(n, 0.0);
  cfg_.watchdog_s = node->get_parameter("watchdog_s").as_double();
  cfg_.soft_margin = node->get_parameter("soft_margin").as_double();
  blend_s_ = node->get_parameter("blend_s").as_double();
  drive_feedback_ = node->get_parameter("drive_feedback").as_bool();
  hold_on_startup_ = node->get_parameter("hold_on_startup").as_bool();
  hold_on_timeout_ = node->get_parameter("hold_on_timeout").as_bool();
  hold_only_ = node->get_parameter("hold_only").as_bool();
  position_only_ = node->get_parameter("position_only").as_bool();
  hold_on_startup_ = hold_on_startup_ || hold_only_ || position_only_;
  hold_on_timeout_ = hold_on_timeout_ || hold_only_ || position_only_;
  gravity_enabled_ = node->get_parameter("gravity_enabled").as_bool() && !hold_only_ && !position_only_;
  if (hold_only_) {
    RCLCPP_WARN(logger, "Relative reference: position holding only; motion commands ignored, "
      "gravity disabled. Joint angles are not registered to the CAD model.");
  }
  local_center_ = node->get_parameter("local_center").as_double_array();
  local_bias_ = node->get_parameter("local_bias").as_double_array();
  local_jacobian_ = node->get_parameter("local_jacobian").as_double_array();
  local_lower_ = node->get_parameter("local_lower").as_double_array();
  local_upper_ = node->get_parameter("local_upper").as_double_array();
  local_gravity_enabled_ = !local_bias_.empty();
  if (local_gravity_enabled_) {
    const auto finite = [](const std::vector<double> & v) {
        return std::all_of(v.begin(), v.end(), [](double x) {return std::isfinite(x);});
      };
    if (!position_only_ || hold_only_ || local_center_.size() != n || local_bias_.size() != n ||
      local_jacobian_.size() != n*n || local_lower_.size() != n || local_upper_.size() != n ||
      !finite(local_center_) || !finite(local_bias_) || !finite(local_jacobian_) ||
      !finite(local_lower_) || !finite(local_upper_))
    {
      RCLCPP_ERROR(logger, "Invalid local gravity parameters");
      return controller_interface::CallbackReturn::ERROR;
    }
    for (size_t i = 0; i < n; ++i) {
      if (!(local_lower_[i] < local_center_[i] && local_center_[i] < local_upper_[i]) ||
        local_lower_[i] < cfg_.lower[i] || local_upper_[i] > cfg_.upper[i])
      {
        RCLCPP_ERROR(logger, "Local gravity domain must be inside the position bounds");
        return controller_interface::CallbackReturn::ERROR;
      }
    }
  }
  write_max_torque_ = node->get_parameter("write_max_torque").as_bool();
  torque_offset_enabled_ = node->get_parameter("torque_offset_enabled").as_bool_array();
  auto gain_ok = [](double g) {return std::isfinite(g) && g >= 0.0 && g <= 1.5;};
  auto tau_ok = [](double t) {return std::isfinite(t) && t >= 0.0 && t <= 0.2;};
  velocity_ff_gain_ = node->get_parameter("velocity_ff_gain").as_double();
  if (!gain_ok(velocity_ff_gain_)) {
    RCLCPP_ERROR(logger, "'velocity_ff_gain' must be in [0, 1.5], got %f", velocity_ff_gain_.load());
    return controller_interface::CallbackReturn::ERROR;
  }
  velocity_ff_tau_s_ = node->get_parameter("velocity_ff_tau_s").as_double();
  if (!tau_ok(velocity_ff_tau_s_)) {
    RCLCPP_ERROR(logger, "'velocity_ff_tau_s' must be in [0, 0.2] s, got %f", velocity_ff_tau_s_.load());
    return controller_interface::CallbackReturn::ERROR;
  }
  vff_.assign(n, 0.0);
  {
    auto valid = [n](const std::vector<double> & v) {
      if (!v.empty() && v.size() != n) {return false;}
      for (double x : v) {if (!std::isfinite(x) || x < 0.0 || x > 2.0) {return false;}}
      return true;
    };
    const auto c = node->get_parameter("coulomb_ff_nm").as_double_array();
    if (!valid(c)) {
      RCLCPP_ERROR(logger, "'coulomb_ff_nm' needs %zu entries in [0, 2] Nm", n);
      return controller_interface::CallbackReturn::ERROR;
    }
    coulomb_ff_ = std::make_unique<std::atomic<double>[]>(n);
    for (size_t i = 0; i < n; ++i) {coulomb_ff_[i] = c.empty() ? 0.0 : c[i];}
    coulomb_cb_ = node->add_on_set_parameters_callback(
      [this, n, valid, gain_ok, tau_ok](const std::vector<rclcpp::Parameter> & ps) {
        rcl_interfaces::msg::SetParametersResult r;
        r.successful = true;
        for (const auto & p : ps) {
          if (p.get_name() == "velocity_ff_gain") {
            if (!gain_ok(p.as_double())) {r.successful = false; r.reason = "velocity_ff_gain: [0, 1.5]"; return r;}
            velocity_ff_gain_ = p.as_double();
            continue;
          }
          if (p.get_name() == "velocity_ff_tau_s") {
            if (!tau_ok(p.as_double())) {r.successful = false; r.reason = "velocity_ff_tau_s: [0, 0.2] s"; return r;}
            velocity_ff_tau_s_ = p.as_double();
            continue;
          }
          if (p.get_name() != "coulomb_ff_nm") {continue;}
          const auto v = p.as_double_array();
          if (!valid(v)) {r.successful = false; r.reason = "coulomb_ff_nm: n entries in [0, 2] Nm"; return r;}
          for (size_t i = 0; i < n; ++i) {coulomb_ff_[i] = v.empty() ? 0.0 : v[i];}
        }
        return r;
      });
  }
  if (torque_offset_enabled_.empty()) {torque_offset_enabled_.assign(n, true);}
  if (torque_offset_enabled_.size() != n) {
    RCLCPP_ERROR(logger, "'torque_offset_enabled' needs %zu entries, got %zu", n,
      torque_offset_enabled_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!torque_offset_enabled_[i]) {
      RCLCPP_WARN(logger, "%s: legacy PDOs; CSP uses the drive position loop without torque "
        "feedforward, CST uses 0x6071", joints_[i].c_str());
    }
  }

  std::string urdf = node->get_parameter("robot_description").as_string();
  if (urdf.empty()) {
    RCLCPP_ERROR(logger, "'robot_description' parameter is empty (the launch file sets it)");
    return controller_interface::CallbackReturn::ERROR;
  }
  gravity_ = std::make_unique<GravityModel>();
  std::string error;
  if (!gravity_->build(
      urdf, joints_, node->get_parameter("calibration_file").as_string(),
      node->get_parameter("link_mass_names").as_string_array(),
      node->get_parameter("link_mass_values").as_double_array(), cfg_, error))
  {
    RCLCPP_ERROR(logger, "%s", error.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  const std::string payload_frame = node->get_parameter("payload_frame").as_string();
  payload_ok_ = gravity_->set_payload_frame(payload_frame);
  if (!payload_ok_) {
    RCLCPP_WARN(logger, "payload_frame '%s' is not in robot_description: payload blocks are ignored",
      payload_frame.c_str());
  }

  q_.assign(n, 0.0);
  q_hold_.assign(n, 0.0);
  qd_.assign(n, 0.0);
  tau_meas_.assign(n, 0.0);
  g_.assign(n, 0.0);
  tau_switch_.assign(n, 0.0);
  q_target_prev_.assign(n, 0.0);
  q_switch_.assign(n, 0.0);
  since_ease_.assign(n, 0.0);
  enabled_.assign(n, true);
  out_.q_target.assign(n, 0.0);
  out_.tau.assign(n, 0.0);

  const size_t expected = kCommandHeader + kCommandBlocks * n;
  subscription_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    "~/command", rclcpp::SystemDefaultsQoS(),
    [this, n, expected](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      if (hold_only_) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Ignoring motion command: relative reference permits position holding only");
        return;
      }
      if (msg->data.size() != expected && msg->data.size() != expected + kPayloadBlock) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "ignoring command with %zu values; expected %zu (or %zu with a payload)", msg->data.size(),
          expected, expected + kPayloadBlock);
        return;
      }
      auto c = std::make_shared<Command>();
      const auto & d = msg->data;
      c->law.present = true;
      c->law.mode = d[0] > 0.5 ? arm::kModeT : arm::kModeP;
      const int flags = std::isfinite(d[1]) ? static_cast<int>(d[1]) : 1;
      c->law.gravity_ff = flags & 1;
      c->law.ignore_soft_limits = flags & 2;
      if (position_only_ && c->law.mode != arm::kModeP) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Relative calibration only permits bounded position commands");
        return;
      }
      if (position_only_) {c->law.ignore_soft_limits = false;}
      c->law.friction_ff_gain = std::isfinite(d[2]) ? d[2] : 1.0;
      auto block = [&](size_t b) {
          return std::vector<double>(d.begin() + kCommandHeader + b * n,
            d.begin() + kCommandHeader + (b + 1) * n);
        };
      c->law.q = block(0);
      c->law.qd = block(1);
      c->law.kp = block(2);
      c->law.kd = block(3);
      c->law.tau_ff = block(4);
      if (position_only_) {std::fill(c->law.tau_ff.begin(), c->law.tau_ff.end(), 0.0);}
      c->law.torque_clamp = block(5);
      if (d.size() == expected + kPayloadBlock && std::isfinite(d[expected])) {
        c->has_payload = true;
        for (size_t k = 0; k < kPayloadBlock; ++k) {
          c->payload[k] = std::isfinite(d[expected + k]) ? d[expected + k] : 0.0;
        }
      }
      c->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(c);
    });
  torque_compensation_ = node->get_parameter("torque_compensation").as_bool();
  joint_position_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    "~/joint_position", rclcpp::SystemDefaultsQoS(),
    [this, n](const sensor_msgs::msg::JointState::SharedPtr msg) {
      const auto log = get_node()->get_logger();
      if (hold_only_) {
        RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
          "Ignoring joint_position: relative reference permits position holding only");
        return;
      }
      auto c = std::make_shared<Command>();
      auto & law = c->law;
      if (!from_joint_state(*msg, msg->position, true, "position", law.q) ||
        !from_joint_state(*msg, msg->velocity, false, "velocity", law.qd) ||
        !from_joint_state(*msg, msg->effort, false, "effort", law.tau_ff))
      {
        return;
      }
      law.present = true;
      law.mode = arm::kModeP;
      law.gravity_ff = true;
      law.friction_ff_gain = 1.0;
      law.kp.assign(n, std::numeric_limits<double>::quiet_NaN());
      law.kd = law.kp;
      law.torque_clamp = law.kp;
      if (position_only_) {std::fill(law.tau_ff.begin(), law.tau_ff.end(), 0.0);}
      c->q_base = law.q;
      c->extrapolate = std::any_of(law.qd.begin(), law.qd.end(), [](double v) {return std::isfinite(v);});
      c->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(c);
    });
  joint_torque_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    "~/joint_torque", rclcpp::SystemDefaultsQoS(),
    [this, n](const sensor_msgs::msg::JointState::SharedPtr msg) {
      const auto log = get_node()->get_logger();
      if (hold_only_ || position_only_) {
        RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
          "Ignoring joint_torque: this calibration only permits position commands");
        return;
      }
      auto c = std::make_shared<Command>();
      auto & law = c->law;
      if (!from_joint_state(*msg, msg->effort, true, "effort", law.tau_ff)) {return;}
      law.present = true;
      law.mode = arm::kModeT;
      law.gravity_ff = torque_compensation_;
      law.friction_ff_gain = 1.0;
      // optional per joint: position -> impedance hold there (config kp/kd),
      // velocity -> damping toward it (config kd). NaN / absent = pure torque.
      const double nan = std::numeric_limits<double>::quiet_NaN();
      law.q.assign(n, nan);
      law.qd = law.kp = law.kd = law.torque_clamp = law.q;
      if (!from_joint_state(*msg, msg->position, false, "position", law.q)) {return;}
      if (!from_joint_state(*msg, msg->velocity, false, "velocity", law.qd)) {return;}
      for (size_t i = 0; i < n; ++i) {
        const bool hold = std::isfinite(law.q[i]);
        if (hold) {law.kp[i] = kp_cfg_[i];}
        if (hold || std::isfinite(law.qd[i])) {law.kd[i] = kd_cfg_[i];}
      }
      c->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(c);
    });
  joint_state_pub_ = node->create_publisher<sensor_msgs::msg::JointState>(
    "~/joint_state", rclcpp::SystemDefaultsQoS());
  rt_joint_state_ = std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(
    joint_state_pub_);
  rt_joint_state_->msg_.name = joints_;
  rt_joint_state_->msg_.position.assign(n, 0.0);
  rt_joint_state_->msg_.velocity.assign(n, 0.0);
  rt_joint_state_->msg_.effort.assign(n, 0.0);

  goto_q0_.assign(n, 0.0);
  goto_c_.assign(6 * n, 0.0);
  goto_law_ = arm::LawCommand();
  goto_law_.present = true;
  goto_law_.age_s = 0.0;
  goto_law_.mode = arm::kModeP;
  goto_law_.q.assign(n, 0.0);
  goto_law_.qd.assign(n, 0.0);
  for (auto * v : {&goto_law_.kp, &goto_law_.kd, &goto_law_.tau_ff, &goto_law_.torque_clamp}) {
    v->assign(n, std::numeric_limits<double>::quiet_NaN());
  }
  goto_sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    "~/goto", rclcpp::SystemDefaultsQoS(),
    [this, n](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      const auto & d = msg->data;
      const auto log = get_node()->get_logger();
      if (hold_only_) {
        RCLCPP_WARN(log, "Ignoring goto: relative reference permits position holding only");
        return;
      }
      const size_t k = d.size() / (n + 1);
      bool ok = k > 0 && d.size() == k * (n + 1) &&
        std::all_of(d.begin(), d.end(), [](double v) {return std::isfinite(v);});
      for (size_t j = 0; ok && j < k; ++j) {ok = d[j * (n + 1)] > 0.0;}
      if (!ok) {
        RCLCPP_WARN(log, "ignoring goto: need [max_speed > 0, q[%zu]] x k, all finite, got %zu values",
          n, d.size());
        return;
      }
      auto g = std::make_shared<Goto>();
      double vmax = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < n; ++i) {vmax = std::min(vmax, cfg_.max_velocity[i]);}
      for (size_t j = 0; j < k; ++j) {
        g->speed.push_back(std::min(d[j * (n + 1)], vmax));
        g->q.insert(g->q.end(), d.begin() + j * (n + 1) + 1, d.begin() + (j + 1) * (n + 1));
      }
      g->T.assign(k, 0.0);
      for (size_t j = 1; j < k; ++j) {
        double dist = 0.0;
        for (size_t i = 0; i < n; ++i) {dist = std::max(dist, std::fabs(g->q[j * n + i] - g->q[(j - 1) * n + i]));}
        g->T[j] = std::max(0.5, 1.875 * dist / g->speed[j]);
      }
      g->received_ns = steady_now_ns();
      goto_buffer_.writeFromNonRT(g);
    });
  state_pub_ = node->create_publisher<std_msgs::msg::Float64MultiArray>(
    "~/state", rclcpp::SystemDefaultsQoS());
  rt_state_ = std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(
    state_pub_);
  rt_state_->msg_.data.assign(kStateHeader + (kStateBlocks + 2) * n, 0.0);

  RCLCPP_INFO(logger, "%zu joints, watchdog %.0f ms; Mode P (CSP) / Mode T (CST)%s", n,
    cfg_.watchdog_s * 1e3, drive_feedback_ ? "; drive feedback (0x6041, 0x6061)" : "");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration ArmController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const char * iface : {"position", "effort", "torque_offset", "velocity_offset",
      "mode_of_operation"})
  {
    for (size_t i = 0; i < joints_.size(); ++i) {
      const std::string name(iface);
      if ((name != "torque_offset" && name != "velocity_offset") || torque_offset_enabled_[i]) {
        config.names.push_back(joints_[i] + "/" + iface);
      }
    }
  }
  if (write_max_torque_) {
    for (const auto & j : joints_) {config.names.push_back(j + "/max_torque");}
  }
  return config;
}

controller_interface::InterfaceConfiguration ArmController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const char * iface : {"position", "velocity", "effort"}) {
    for (const auto & j : joints_) {config.names.push_back(j + "/" + iface);}
  }
  if (drive_feedback_) {
    for (const char * iface : {"status_word", "mode_of_operation"}) {
      for (const auto & j : joints_) {config.names.push_back(j + "/" + iface);}
    }
  }
  return config;
}

controller_interface::CallbackReturn ArmController::on_activate(const rclcpp_lifecycle::State &)
{
  cmd_pos_.clear(); cmd_eff_.clear(); cmd_off_.clear(); cmd_mode_.clear(); cmd_max_.clear();
  st_pos_.clear(); st_vel_.clear(); st_eff_.clear(); st_status_.clear(); st_mode_.clear();
  bool ok =
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "position", cmd_pos_) &&
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "effort", cmd_eff_) &&
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "mode_of_operation", cmd_mode_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "position", st_pos_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "velocity", st_vel_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "effort", st_eff_);
  cmd_off_.assign(joints_.size(), nullptr);
  cmd_voff_.assign(joints_.size(), nullptr);
  for (size_t i = 0; ok && i < joints_.size(); ++i) {
    if (!torque_offset_enabled_[i]) {continue;}
    for (auto * slot : {&cmd_off_[i], &cmd_voff_[i]}) {
      const auto name = joints_[i] + (slot == &cmd_off_[i] ? "/torque_offset" : "/velocity_offset");
      const auto iface = std::find_if(command_interfaces_.begin(), command_interfaces_.end(),
        [&](const auto & candidate) {return candidate.get_name() == name;});
      if (iface == command_interfaces_.end()) {
        ok = false;
        break;
      }
      *slot = &*iface;
    }
  }
  if (ok && write_max_torque_) {
    ok = controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "max_torque", cmd_max_);
  }
  if (ok && drive_feedback_) {
    ok = controller_interface::get_ordered_interfaces(
      state_interfaces_, joints_, "status_word", st_status_) &&
      controller_interface::get_ordered_interfaces(
      state_interfaces_, joints_, "mode_of_operation", st_mode_);
  }
  if (!ok) {
    RCLCPP_ERROR(get_node()->get_logger(), "could not get the expected interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  // Start held in CSP at the actual position; no command -> watchdog fallback
  // next cycle (gravity comp + damping) until Python sends targets.
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  goto_buffer_.writeFromNonRT(std::shared_ptr<Goto>());
  goto_ns_ = 0;
  goto_cancelled_ns_ = 0;
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_target_prev_[i] = st_pos_[i].get().get_value();
    cmd_pos_[i].get().set_value(q_target_prev_[i]);
    cmd_eff_[i].get().set_value(0.0);
    if (cmd_off_[i]) {cmd_off_[i]->set_value(0.0);}
    if (cmd_voff_[i]) {cmd_voff_[i]->set_value(0.0);}
    vff_[i] = 0.0;
    cmd_mode_[i].get().set_value(static_cast<double>(arm::kCsp));
    if (write_max_torque_) {cmd_max_[i].get().set_value(cfg_.max_torque[i]);}
  }
  last_mode_ = arm::kModeP;
  holding_ = false;
  since_switch_ = 0.0;
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_switch_[i] = q_target_prev_[i];
    since_ease_[i] = 0.0;
  }
  std::fill(enabled_.begin(), enabled_.end(), true);
  std::fill(std::begin(payload_), std::end(payload_), 0.0);
  gravity_->set_payload(0.0, 0.0, 0.0, 0.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ArmController::on_deactivate(const rclcpp_lifecycle::State &)
{
  // Hold position in CSP with no feedforward.
  for (size_t i = 0; i < cmd_pos_.size(); ++i) {
    cmd_pos_[i].get().set_value(st_pos_[i].get().get_value());
    cmd_eff_[i].get().set_value(0.0);
    if (cmd_off_[i]) {cmd_off_[i]->set_value(0.0);}
    if (cmd_voff_[i]) {cmd_voff_[i]->set_value(0.0);}
    cmd_mode_[i].get().set_value(static_cast<double>(arm::kCsp));
  }
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  goto_buffer_.writeFromNonRT(std::shared_ptr<Goto>());
  return controller_interface::CallbackReturn::SUCCESS;
}

bool ArmController::from_joint_state(
  const sensor_msgs::msg::JointState & msg, const std::vector<double> & field,
  bool required, const char * what, std::vector<double> & out) const
{
  const size_t n = joints_.size();
  const auto log = get_node()->get_logger();
  out.assign(n, std::numeric_limits<double>::quiet_NaN());
  if (field.empty()) {
    if (required) {
      RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000, "ignoring JointState: no %s", what);
    }
    return !required;
  }
  if (msg.name.empty()) {
    if (field.size() != n) {
      RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
        "ignoring JointState: %zu %s values without names; expected %zu (in 'joints' order)",
        field.size(), what, n);
      return false;
    }
    out = field;
  } else {
    if (field.size() != msg.name.size()) {
      RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
        "ignoring JointState: %zu names but %zu %s values", msg.name.size(), field.size(), what);
      return false;
    }
    for (size_t i = 0; i < n; ++i) {
      const auto it = std::find(msg.name.begin(), msg.name.end(), joints_[i]);
      if (it == msg.name.end()) {
        RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
          "ignoring JointState: no %s for joint '%s'", what, joints_[i].c_str());
        return false;
      }
      out[i] = field[static_cast<size_t>(it - msg.name.begin())];
    }
  }
  if (required && !std::all_of(out.begin(), out.end(), [](double v) {return std::isfinite(v);})) {
    RCLCPP_WARN_THROTTLE(log, *get_node()->get_clock(), 2000,
      "ignoring JointState: a %s value is not finite", what);
    return false;
  }
  return true;
}

// Goto waypoint j for joint i; j = -1 is where the path started.
double ArmController::goto_wp(const Goto & g, long j, size_t i) const
{
  return j < 0 ? goto_q0_[i] : g.q[static_cast<size_t>(j) * joints_.size() + i];
}

// Velocity and acceleration of joint i at waypoint j (the end of segment j). The start
// and the last waypoint are at rest. A joint still on either side comes to rest; one
// that keeps going the same way passes at the mean of the two segment speeds (capped); one
// that reverses stops with the acceleration a smooth reversal has (the mean of the
// two segments' cubic end accelerations), so it turns without a dwell.
void ArmController::goto_via(const Goto & g, long j, size_t i, double & v, double & a) const
{
  v = a = 0.0;
  if (j < 0 || static_cast<size_t>(j) + 1 >= g.k()) {return;}
  const double d1 = goto_wp(g, j, i) - goto_wp(g, j - 1, i);
  const double d2 = goto_wp(g, j + 1, i) - goto_wp(g, j, i);
  const double T1 = goto_seg_T(g, j), T2 = goto_seg_T(g, j + 1);
  if (std::fabs(d1) < 1e-6 || std::fabs(d2) < 1e-6) {return;}
  if (d1 * d2 > 0.0) {
    // capped at twice the slower segment's mean speed: more and the quintic on
    // that side backs up before it moves (overshoots its start)
    const double m1 = d1 / T1, m2 = d2 / T2;
    const double cap = 2.0 * std::min(std::fabs(m1), std::fabs(m2));
    v = std::copysign(std::min(0.5 * std::fabs(m1 + m2), cap), d1);
  } else {
    a = 0.5 * (-6.0 * d1 / (T1 * T1) + 6.0 * d2 / (T2 * T2));
  }
}

// Quintic for the current segment from its end conditions (all zero: min-jerk).
void ArmController::goto_plan_segment(const Goto & g)
{
  const long j = static_cast<long>(goto_seg_);
  const double T = goto_seg_T(g, goto_seg_);
  for (size_t i = 0; i < joints_.size(); ++i) {
    const double p0 = goto_wp(g, j - 1, i), p1 = goto_wp(g, j, i);
    double v0, a0, v1, a1;
    goto_via(g, j - 1, i, v0, a0);
    goto_via(g, j, i, v1, a1);
    const double dp = p1 - p0, T2 = T * T, T3 = T2 * T;
    double * c = &goto_c_[6 * i];
    c[0] = p0;
    c[1] = v0;
    c[2] = 0.5 * a0;
    c[3] = (20.0 * dp - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T2) / (2.0 * T3);
    c[4] = (-30.0 * dp + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T2) / (2.0 * T3 * T);
    c[5] = (12.0 * dp - 6.0 * (v1 + v0) * T - (a0 - a1) * T2) / (2.0 * T3 * T2);
  }
}

controller_interface::return_type ArmController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  const size_t n = joints_.size();
  const int64_t now = steady_now_ns();   // command age (the watchdog) only
  // Blends and eases count control periods, not wall time: the drives step
  // once per bus cycle, so a late cycle must not skip part of a ramp.
  const double dt = std::max(period.seconds(), 1e-4);
  since_switch_ += dt;
  for (size_t i = 0; i < n; ++i) {
    since_ease_[i] += dt;
    q_[i] = st_pos_[i].get().get_value();
    qd_[i] = st_vel_[i].get().get_value();
    tau_meas_[i] = st_eff_[i].get().get_value();
  }

  // A drive that is not OPERATION_ENABLED follows nothing: hold the whole arm
  // in the fallback until every drive is back.
  bool all_enabled = true;
  if (drive_feedback_) {
    for (size_t i = 0; i < n; ++i) {
      const double sw = st_status_[i].get().get_value();
      enabled_[i] = std::isfinite(sw) && (static_cast<uint16_t>(sw) & 0x6F) == 0x27;
      all_enabled = all_enabled && enabled_[i];
    }
  }

  static const arm::LawCommand kNone;
  const std::shared_ptr<Command> cmd = *command_buffer_.readFromRT();
  if (cmd) {
    cmd->law.age_s = (now - cmd->received_ns) * 1e-9;
    if (cmd->extrapolate) {
      const double h = std::clamp(cmd->law.age_s, 0.0, cfg_.watchdog_s);
      for (size_t i = 0; i < n; ++i) {
        const double v = cmd->law.qd[i];
        cmd->law.q[i] = cmd->q_base[i] + (std::isfinite(v) ? v * h : 0.0);
      }
    }
    // the payload is sticky: kept after the command goes stale (the watchdog
    // fallback still carries it) and by later commands without the block
    if (cmd->has_payload && payload_ok_ &&
      !std::equal(std::begin(payload_), std::end(payload_), std::begin(cmd->payload)))
    {
      std::copy(std::begin(cmd->payload), std::end(cmd->payload), std::begin(payload_));
      gravity_->set_payload(payload_[0], payload_[1], payload_[2], payload_[3]);
    }
  }
  if (local_gravity_enabled_) {
    // The local fit is only trusted inside its box, so it is never extrapolated:
    // outside, each joint's offset from the centre saturates at the box edge. That
    // keeps g continuous (a step to 0 at the edge would drop the shoulder) and
    // bounded by its value on the box. A non-finite reading turns it off.
    bool finite_q = true;
    for (size_t i = 0; i < n; ++i) {finite_q = finite_q && std::isfinite(q_[i]);}
    for (size_t i = 0; i < n; ++i) {
      g_[i] = finite_q ? local_bias_[i] : 0.0;
      if (finite_q) {
        for (size_t j = 0; j < n; ++j) {
          const double qj = std::clamp(q_[j], local_lower_[j], local_upper_[j]);
          g_[i] += local_jacobian_[i*n+j]*(qj-local_center_[j]);
        }
      }
    }
  } else if (gravity_enabled_) {gravity_->gravity(q_, g_);} else {std::fill(g_.begin(), g_.end(), 0.0);}
  // A goto newer than the last streamed command runs the min-jerk path here, one
  // step per control period, and then keeps holding its goal.
  const arm::LawCommand * law = cmd ? &cmd->law : nullptr;
  const std::shared_ptr<Goto> go = *goto_buffer_.readFromRT();
  if (go && go->received_ns != goto_cancelled_ns_ && (!cmd || go->received_ns > cmd->received_ns)) {
    if (!all_enabled) {
      goto_cancelled_ns_ = go->received_ns;   // a drive dropped out: do not resume a stale path
      goto_ns_ = 0;
    } else {
      if (go->received_ns != goto_ns_) {
        goto_ns_ = go->received_ns;
        goto_t_ = 0.0;
        double dist = 0.0;
        for (size_t i = 0; i < n; ++i) {
          // from the last CSP target (continuous), or the actual angle out of Mode T
          goto_q0_[i] = last_mode_ == arm::kModeP && !holding_ ? q_target_prev_[i] : q_[i];
          if (holding_) {goto_q0_[i] = q_hold_[i];}
          dist = std::max(dist, std::fabs(go->q[i] - goto_q0_[i]));
        }
        goto_T0_ = std::max(0.5, 1.875 * dist / go->speed[0]);
        goto_seg_ = 0;
        goto_plan_segment(*go);
      } else {
        goto_t_ += dt;
        while (goto_t_ >= goto_seg_T(*go, goto_seg_) && goto_seg_ + 1 < go->k()) {
          goto_t_ -= goto_seg_T(*go, goto_seg_);
          ++goto_seg_;
          goto_plan_segment(*go);
        }
      }
      const double t = std::min(goto_t_, goto_seg_T(*go, goto_seg_));
      for (size_t i = 0; i < n; ++i) {
        const double * c = &goto_c_[6 * i];
        goto_law_.q[i] = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
        goto_law_.qd[i] = c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])));
      }
      law = &goto_law_;
    }
  }
  arm::compute(cfg_, law && all_enabled ? *law : kNone, q_, qd_, g_, out_);
  const bool hold = out_.fallback && (hold_on_timeout_ || (hold_on_startup_ && !cmd));
  if (hold) {
    if (!holding_ || ((hold_only_ || position_only_) && !all_enabled)) {q_hold_ = q_;}
    // A drive that is not enabled is not held where the hold began: it may have
    // dropped. Follow it, so re-enabling holds it where it is instead of driving
    // it back to a stale angle.
    for (size_t i = 0; i < n; ++i) {
      if (!enabled_[i]) {q_hold_[i] = q_[i];}
    }
    out_.mode = arm::kModeP;
    out_.q_target = q_hold_;
    // Keep the gravity feedforward (0 when gravity is off, e.g. hold_only). With
    // tau = 0 a drive still in CST during the CST -> CSP switch would get no torque
    // at all (0x6071 = 0x60B2 = 0) and the arm would drop, and once in CSP the
    // position loop would hold it with no gravity support.
    for (size_t i = 0; i < n; ++i) {
      out_.tau[i] = std::clamp(g_[i], -cfg_.max_torque[i], cfg_.max_torque[i]);
    }
  }
  holding_ = hold;

  if (out_.mode != last_mode_) {
    since_switch_ = 0.0;
    for (size_t i = 0; i < n; ++i) {
      tau_switch_[i] = tau_meas_[i];
      q_target_prev_[i] = q_[i];   // CSP re-entry starts at the actual position
      q_switch_[i] = q_[i];
      since_ease_[i] = 0.0;
    }
    last_mode_ = out_.mode;
  }
  for (size_t i = 0; i < n; ++i) {
    const bool mode_p = out_.mode == arm::kModeP;
    const int requested = mode_p ? arm::kCsp : arm::kCst;
    bool pending = false;   // drive has not confirmed the requested mode yet
    if (drive_feedback_) {
      const double shown = st_mode_[i].get().get_value();
      pending = (shown == arm::kCsp || shown == arm::kCst) && shown != requested;
    }
    if (!enabled_[i] || (mode_p && pending)) {
      // Not following a CSP target yet (disabled, or the master still sends
      // the actual position as 0x607A): start, and keep restarting, the
      // ease-in from where the joint is, so it never resumes at a stale target.
      q_target_prev_[i] = q_[i];
      q_switch_[i] = q_[i];
      since_ease_[i] = 0.0;
    }
    if (!enabled_[i]) {
      // Re-enabling goes straight to the law's torque: the measured torque is
      // 0 while disabled, and blending up from 0 would let the arm drop.
      tau_switch_[i] = out_.tau[i];
    }
    double target = q_[i];   // Mode T: keeps CSP re-entry bumpless
    double tau = out_.tau[i];
    // Velocity feedforward: the CSP target's own velocity, so the drive moves with
    // the target instead of waiting for position error to beat the gearbox
    // friction (stick-slip). Derived from the target actually sent: 0 while holding,
    // and it follows the ease-in and rate limit. 20 ms low-pass for the uneven
    // spacing of the Python setpoints.
    double vff = 0.0;
    if (mode_p) {
      const double goal =
        arm::ease_in(q_switch_[i], out_.q_target[i], since_ease_[i], blend_s_);
      target = arm::rate_limit(q_target_prev_[i], goal, cfg_.max_velocity[i] * dt);
      const double alpha = dt / (dt + velocity_ff_tau_s_.load(std::memory_order_relaxed));
      vff_[i] += alpha * ((target - q_target_prev_[i]) / dt - vff_[i]);
      // A goto knows its exact path velocity: use it (no differentiation, no filter
      // lag), as long as the target sent IS that path (not clamped by a limit or
      // keep-out, not easing in, not rate-limited). Streamed commands keep the filter.
      double v_ref = vff_[i];
      if (law == &goto_law_ && out_.q_target[i] == goto_law_.q[i] && since_ease_[i] >= blend_s_ &&
        target == goal)
      {
        v_ref = goto_law_.qd[i];
      }
      vff = std::clamp(velocity_ff_gain_.load(std::memory_order_relaxed) * v_ref, -cfg_.max_velocity[i], cfg_.max_velocity[i]);
      q_target_prev_[i] = target;
    } else {
      vff_[i] = 0.0;
    }
    if (!enabled_[i] || pending || hold) {
      vff_[i] = 0.0;
      vff = 0.0;
    }
    if (cmd_voff_[i]) {cmd_voff_[i]->set_value(vff);}
    if (!mode_p) {
      tau = arm::blend(tau_switch_[i], out_.tau[i], since_switch_, blend_s_);
      out_.tau[i] = tau;
    }
    // Torque channel: 0x60B2 offset in CSP, 0x6071 target in CST. While a
    // switch is pending, both, so the torque is applied whichever mode the
    // drive runs this cycle (CSP ignores 0x6071; the RMD-X ignores 0x60B2 in CST).
    cmd_mode_[i].get().set_value(static_cast<double>(requested));
    cmd_pos_[i].get().set_value(target);
    double cap = cfg_.max_torque[i];
    if (law && !out_.fallback && i < law->torque_clamp.size() &&
      std::isfinite(law->torque_clamp[i]))
    {
      cap = std::min(cap, std::fabs(law->torque_clamp[i]));
    }
    if (write_max_torque_) {
      // 0x6072 also caps the drive's own CSP position loop, e.g. during a
      // limit search pushing into a hard stop.
      // 0x6072 = 0 is NOT zero torque on the RMD-X: the drive falls back to its
      // stall-current limit. Never let a zero (or tiny, rounding to 0 per mille)
      // clamp reach the drive: floor it at 1% of the configured max.
      cmd_max_[i].get().set_value(std::max(cap, kMinTorqueCapFraction * cfg_.max_torque[i]));
    }
    if (mode_p && vff != 0.0 && coulomb_ff_) {
      // break the gearbox's static friction with the target, not after the error builds
      tau = std::clamp(tau + coulomb_ff_[i].load(std::memory_order_relaxed) * std::tanh(vff / 0.02),
          -cap, cap);
    }
    if (!cmd_off_[i]) {
      // 0x6071 is ignored in CSP. Keep the measured switching torque there
      // during a delayed CST -> CSP transition, then the drive's position
      // loop takes over. No torque-offset object is needed on legacy drives.
      const double effort = mode_p ? std::clamp(tau_switch_[i], -cap, cap) : tau;
      cmd_eff_[i].get().set_value(effort);
      out_.tau[i] = mode_p ? 0.0 : tau;
    } else if (pending) {
      // The drive switches before its 0x6061 shows it, and the RMD-X does not add
      // 0x60B2 in CST: with 0x6071 = 0 here it ran CST on zero torque for ~10 ms
      // (2026-11-23: J2 2.4 Nm commanded, 0.1 Nm measured, the arm dropped at every
      // CSP -> CST switch). Send the torque on both channels until the mode shows.
      cmd_off_[i]->set_value(tau);
      cmd_eff_[i].get().set_value(tau);
    } else if (requested == arm::kCsp) {
      cmd_off_[i]->set_value(tau);
      cmd_eff_[i].get().set_value(0.0);
    } else {
      cmd_eff_[i].get().set_value(tau);
      cmd_off_[i]->set_value(0.0);
    }
  }

  if (rt_state_ && rt_state_->trylock()) {
    auto & d = rt_state_->msg_.data;
    d[0] = now * 1e-9;
    d[1] = out_.mode;
    d[2] = out_.fallback;
    d[3] = out_.limit_hit;
    d[4] = out_.clamped;
    for (size_t i = 0; i < n; ++i) {
      d[kStateHeader + i] = q_[i];
      d[kStateHeader + n + i] = qd_[i];
      d[kStateHeader + 2 * n + i] = tau_meas_[i];
      d[kStateHeader + 3 * n + i] = out_.tau[i];
      d[kStateHeader + 4 * n + i] = out_.mode == arm::kModeP ? q_target_prev_[i] : q_[i];
      d[kStateHeader + 5 * n + i] = drive_feedback_ ? st_status_[i].get().get_value() :
        std::numeric_limits<double>::quiet_NaN();
      d[kStateHeader + 6 * n + i] = drive_feedback_ ? st_mode_[i].get().get_value() :
        std::numeric_limits<double>::quiet_NaN();
    }
    rt_state_->unlockAndPublish();
  }
  if (rt_joint_state_ && rt_joint_state_->trylock()) {
    auto & m = rt_joint_state_->msg_;
    m.header.stamp = time;
    for (size_t i = 0; i < n; ++i) {
      m.position[i] = q_[i];
      m.velocity[i] = qd_[i];
      m.effort[i] = tau_meas_[i];
    }
    rt_joint_state_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(wisco_ethercat_ros2_hw::ArmController, controller_interface::ControllerInterface)
