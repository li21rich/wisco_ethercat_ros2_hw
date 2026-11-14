#include "wisco_ethercat_ros2_hw/arm_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

#include "controller_interface/helpers.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "yaml-cpp/yaml.h"

namespace wisco_ethercat_ros2_hw
{

// Gravity torques from the URDF, with the calibration layered on top the same
// way arm_stack/model/robot.py does it: weighed masses and fitted COMs per
// link (stage 5) and the mocap kinematic overlay. Joints not driven (the
// wrist parallelogram's mimic joints) are locked at zero.
class GravityModel
{
public:
  bool build(
    const std::string & urdf_xml, const std::vector<std::string> & joints,
    const std::string & calibration_file, const std::vector<std::string> & mass_links,
    const std::vector<double> & mass_values, arm::LawConfig & cfg, std::string & error)
  {
    pinocchio::Model full;
    try {
      pinocchio::urdf::buildModelFromXML(urdf_xml, full);
    } catch (const std::exception & e) {
      error = std::string("cannot parse robot_description: ") + e.what();
      return false;
    }
    std::vector<pinocchio::JointIndex> lock;
    for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(full.njoints); ++j) {
      if (std::find(joints.begin(), joints.end(), full.names[j]) == joints.end()) {
        lock.push_back(j);
      }
    }
    model_ = lock.empty() ? full :
      pinocchio::buildReducedModel(full, lock, pinocchio::neutral(full));
    model_.gravity.linear(Eigen::Vector3d(0, 0, -9.81));
    for (const auto & name : joints) {
      if (!model_.existJointName(name)) {
        error = "joint '" + name + "' is not in robot_description";
        return false;
      }
      const auto id = model_.getJointId(name);
      jid_.push_back(id);
      vidx_.push_back(model_.joints[id].idx_v());
    }
    if (static_cast<size_t>(model_.nv) != joints.size()) {
      error = "robot_description has " + std::to_string(model_.nv) + " driven dofs, config lists " +
        std::to_string(joints.size());
      return false;
    }
    // child link of each joint, for the per-link calibration fields
    std::map<std::string, pinocchio::JointIndex> link_joint;
    for (const auto & f : model_.frames) {
      if (f.type == pinocchio::BODY) {link_joint[f.name] = f.parentJoint;}
    }
    auto set_mass = [&](const std::string & link, double mass, const Eigen::Vector3d * com) {
        auto it = link_joint.find(link);
        if (it == link_joint.end() || !(mass > 0)) {return;}
        auto & I = model_.inertias[it->second];
        const double scale = I.mass() > 0 ? mass / I.mass() : 1.0;
        I = pinocchio::Inertia(mass, com ? *com : I.lever(), I.inertia().matrix() * scale);
      };
    for (size_t k = 0; k < mass_links.size() && k < mass_values.size(); ++k) {
      if (std::isfinite(mass_values[k])) {set_mass(mass_links[k], mass_values[k], nullptr);}
    }
    if (!calibration_file.empty()) {
      try {
        const YAML::Node cal = YAML::LoadFile(calibration_file);
        if (cal["link_inertials"] && cal["link_inertials"].IsMap()) {
          for (const auto & kv : cal["link_inertials"]) {
            const auto link = kv.first.as<std::string>();
            const auto & e = kv.second;
            const double mass = e["mass_kg"] && !e["mass_kg"].IsNull() ? e["mass_kg"].as<double>() : -1;
            Eigen::Vector3d com;
            bool has_com = e["com_m"] && e["com_m"].IsSequence() && e["com_m"].size() == 3;
            if (has_com) {
              for (int i = 0; i < 3; ++i) {
                if (e["com_m"][i].IsNull()) {has_com = false;} else {com[i] = e["com_m"][i].as<double>();}
              }
            }
            auto it = link_joint.find(link);
            if (it != link_joint.end()) {
              set_mass(link, mass > 0 ? mass : model_.inertias[it->second].mass(), has_com ? &com : nullptr);
            }
          }
        }
        if (cal["friction"] && cal["friction"].IsMap()) {
          for (size_t i = 0; i < joints.size(); ++i) {
            const auto f = cal["friction"][joints[i]];
            if (!f) {continue;}
            if (f["coulomb_nm"] && !f["coulomb_nm"].IsNull()) {cfg.coulomb[i] = f["coulomb_nm"].as<double>();}
            if (f["viscous_nm_s_per_rad"] && !f["viscous_nm_s_per_rad"].IsNull()) {
              cfg.viscous[i] = f["viscous_nm_s_per_rad"].as<double>();
            }
          }
        }
        if (cal["kinematic_overlay"] && cal["kinematic_overlay"].IsMap()) {
          for (size_t i = 0; i < joints.size(); ++i) {
            const auto o = cal["kinematic_overlay"][joints[i]];
            if (!o) {continue;}
            Eigen::Vector3d dxyz = Eigen::Vector3d::Zero(), drot = Eigen::Vector3d::Zero();
            for (int a = 0; a < 3; ++a) {
              if (o["dxyz"] && o["dxyz"].size() == 3) {dxyz[a] = o["dxyz"][a].as<double>();}
              if (o["drot"] && o["drot"].size() == 3) {drot[a] = o["drot"][a].as<double>();}
            }
            auto & M = model_.jointPlacements[jid_[i]];
            M = pinocchio::SE3(M.rotation() * pinocchio::exp3(drot), M.translation() + dxyz);
          }
        }
      } catch (const std::exception & e) {
        error = "cannot read calibration file '" + calibration_file + "': " + e.what();
        return false;
      }
    }
    data_ = std::make_unique<pinocchio::Data>(model_);
    q_ = pinocchio::neutral(model_);
    return true;
  }

  // Not allocating: q_ and data_ are preallocated.
  void gravity(const std::vector<double> & q, std::vector<double> & g)
  {
    for (size_t i = 0; i < jid_.size(); ++i) {q_[model_.joints[jid_[i]].idx_q()] = q[i];}
    const auto & tau = pinocchio::computeGeneralizedGravity(model_, *data_, q_);
    for (size_t i = 0; i < vidx_.size(); ++i) {g[i] = tau[vidx_[i]];}
  }

private:
  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  Eigen::VectorXd q_;
  std::vector<pinocchio::JointIndex> jid_;
  std::vector<int> vidx_;
};

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
    auto_declare<bool>("write_max_torque", false);
    auto_declare<std::string>("calibration_file", "");
    auto_declare<std::string>("robot_description", "");
    auto_declare<std::vector<std::string>>("link_mass_names", std::vector<std::string>{});
    auto_declare<std::vector<double>>("link_mass_values", std::vector<double>{});
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
  for (size_t i = 0; i < n; ++i) {
    if (!(cfg_.max_torque[i] > 0) || !(cfg_.max_velocity[i] > 0) || !(cfg_.lower[i] < cfg_.upper[i])) {
      RCLCPP_ERROR(logger, "bad limits for joint '%s'", joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  cfg_.coulomb.assign(n, 0.0);
  cfg_.viscous.assign(n, 0.0);
  cfg_.watchdog_s = node->get_parameter("watchdog_s").as_double();
  cfg_.soft_margin = node->get_parameter("soft_margin").as_double();
  blend_s_ = node->get_parameter("blend_s").as_double();
  drive_feedback_ = node->get_parameter("drive_feedback").as_bool();
  write_max_torque_ = node->get_parameter("write_max_torque").as_bool();

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

  q_.assign(n, 0.0);
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
      if (msg->data.size() != expected) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "ignoring command with %zu values; expected %zu", msg->data.size(), expected);
        return;
      }
      auto c = std::make_shared<Command>();
      const auto & d = msg->data;
      c->law.present = true;
      c->law.mode = d[0] > 0.5 ? arm::kModeT : arm::kModeP;
      const int flags = std::isfinite(d[1]) ? static_cast<int>(d[1]) : 1;
      c->law.gravity_ff = flags & 1;
      c->law.ignore_soft_limits = flags & 2;
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
      c->law.torque_clamp = block(5);
      c->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(c);
    });
  state_pub_ = node->create_publisher<std_msgs::msg::Float64MultiArray>(
    "~/state", rclcpp::SystemDefaultsQoS());
  rt_state_ = std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(
    state_pub_);
  rt_state_->msg_.data.assign(kStateHeader + kStateBlocks * n, 0.0);

  RCLCPP_INFO(logger, "%zu joints, watchdog %.0f ms; Mode P (CSP) / Mode T (CST)%s", n,
    cfg_.watchdog_s * 1e3, drive_feedback_ ? "; drive feedback (0x6041, 0x6061)" : "");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration ArmController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const char * iface : {"position", "effort", "torque_offset", "mode_of_operation"}) {
    for (const auto & j : joints_) {config.names.push_back(j + "/" + iface);}
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
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "torque_offset", cmd_off_) &&
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "mode_of_operation", cmd_mode_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "position", st_pos_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "velocity", st_vel_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "effort", st_eff_);
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
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_target_prev_[i] = st_pos_[i].get().get_value();
    cmd_pos_[i].get().set_value(q_target_prev_[i]);
    cmd_eff_[i].get().set_value(0.0);
    cmd_off_[i].get().set_value(0.0);
    cmd_mode_[i].get().set_value(arm::kCsp);
    if (write_max_torque_) {cmd_max_[i].get().set_value(cfg_.max_torque[i]);}
  }
  last_mode_ = arm::kModeP;
  since_switch_ = 0.0;
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_switch_[i] = q_target_prev_[i];
    since_ease_[i] = 0.0;
  }
  std::fill(enabled_.begin(), enabled_.end(), true);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ArmController::on_deactivate(const rclcpp_lifecycle::State &)
{
  // Hold position in CSP with no feedforward.
  for (size_t i = 0; i < cmd_pos_.size(); ++i) {
    cmd_pos_[i].get().set_value(st_pos_[i].get().get_value());
    cmd_eff_[i].get().set_value(0.0);
    cmd_off_[i].get().set_value(0.0);
    cmd_mode_[i].get().set_value(arm::kCsp);
  }
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ArmController::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
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
  gravity_->gravity(q_, g_);

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
  }
  arm::compute(cfg_, cmd && all_enabled ? cmd->law : kNone, q_, qd_, g_, out_);

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
    if (mode_p) {
      const double goal =
        arm::ease_in(q_switch_[i], out_.q_target[i], since_ease_[i], blend_s_);
      target = arm::rate_limit(q_target_prev_[i], goal, cfg_.max_velocity[i] * dt);
      q_target_prev_[i] = target;
    } else {
      tau = arm::blend(tau_switch_[i], out_.tau[i], since_switch_, blend_s_);
      out_.tau[i] = tau;
    }
    // Torque channel: 0x60B2 offset in CSP, 0x6071 target in CST. While a
    // switch is pending, 0x60B2, which CiA402 adds in both modes, so the
    // torque is applied whichever mode the drive runs this cycle.
    cmd_mode_[i].get().set_value(requested);
    cmd_pos_[i].get().set_value(target);
    if (write_max_torque_) {
      // 0x6072 caps the drive's TOTAL torque, so in CSP the drive's own
      // position loop is held to the command's clamp too (a limit search
      // pushing into a hard stop). The fallback uses the configured max.
      double cap = cfg_.max_torque[i];
      if (cmd && !out_.fallback && i < cmd->law.torque_clamp.size() &&
        std::isfinite(cmd->law.torque_clamp[i]))
      {
        cap = std::min(cap, std::fabs(cmd->law.torque_clamp[i]));
      }
      cmd_max_[i].get().set_value(cap);
    }
    if (requested == arm::kCsp || pending) {
      cmd_off_[i].get().set_value(tau);
      cmd_eff_[i].get().set_value(0.0);
    } else {
      cmd_eff_[i].get().set_value(tau);
      cmd_off_[i].get().set_value(0.0);
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
    }
    rt_state_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(wisco_ethercat_ros2_hw::ArmController, controller_interface::ControllerInterface)
