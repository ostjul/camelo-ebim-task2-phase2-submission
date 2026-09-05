// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <franka_fr3_arm_controllers/joint_impedance_controller.hpp>

#include <Eigen/Eigen>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

using std::placeholders::_1;

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  updateJointStates_();
  Vector7d q_goal;
  Vector7d tau_d_calculated;

  if (!motion_generator_initialized_) {
    // After starting the controller we wait for valid joint states from the input topic
    // Until we get valid joint states we will send zero torques to the robot
    // to allow the user to reposition the robot
    motion_generator_initialized_ = initializeMotionGenerator_();

    if (!motion_generator_initialized_) {
      for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(0.0);
      }

      return controller_interface::return_type::OK;
    }
  }

  if (!move_to_start_position_finished_) {
    // We have received valid joint states and initialized the motion generator
    // Now we move smoothly to the first joint position received from the input topic
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;

    q_goal = motion_generator_output.first;
  }

  if (move_to_start_position_finished_) {
    // After reaching the start position we follow the joint position from the input topic
    // This is the normal operation mode of the controller
    if (!hasFreshGelloState_()) {
      RCLCPP_FATAL(get_node()->get_logger(), "Timeout: No valid joint states received from Gello");
      rclcpp::shutdown();  // Exit the node permanently
    }
    q_goal = mapGelloToRobotGoal_();

    // Rate-limit how fast the commanded goal may chase the GELLO. The leader arm can be
    // moved far quicker than the follower may safely track; feeding a distant target
    // straight to the impedance law demands a large correction torque and trips the
    // robot's velocity reflex. Limiting the goal keeps the arm following the same path,
    // just lagging behind a fast hand instead of lunging after it.
    if (max_goal_velocity_ > 0.0) {
      if (!q_goal_limited_valid_) {
        q_goal_limited_ = q_;  // start from where the arm actually is
        q_goal_limited_valid_ = true;
      }
      const double max_step = max_goal_velocity_ * period.seconds();
      for (int i = 0; i < num_joints; ++i) {
        const double delta = q_goal(i) - q_goal_limited_(i);
        q_goal_limited_(i) += std::clamp(delta, -max_step, max_step);
      }
      q_goal = q_goal_limited_;
    }
  }

  tau_d_calculated = calculateTauDGains_(q_goal);

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(const sensor_msgs::msg::JointState msg) {
  const auto receive_time = get_node()->get_clock()->now();
  if (!validateGelloJointState_(msg, receive_time)) {
    gello_position_values_valid_ = false;
    return;
  }

  std::copy(msg.position.begin(), msg.position.end(), gello_position_values_.begin());

  // Keep receiving GELLO targets while this lifecycle controller is inactive.
  // Activation must not be the event that enables input reception: doing so
  // enters torque mode before a valid target exists and only receives the first
  // target a few control cycles later.
  last_valid_gello_receive_time_ = receive_time;
  gello_position_values_valid_ = true;
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<std::string>>("gello_joint_names", {});
    auto_declare<double>("future_timestamp_tolerance", 0.05);
    auto_declare<std::vector<double>>("gello_joint_directions",
                                      std::vector<double>(num_joints, 1.0));
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("motion_generator_speed_factor", 0.05);
    auto_declare<double>("max_goal_velocity", 0.5);
    auto_declare<double>("max_gello_message_age", 0.5);
    auto_declare<double>("max_gello_liveness_gap", 2.0);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto configured_gello_joint_names =
      get_node()->get_parameter("gello_joint_names").as_string_array();
  if (configured_gello_joint_names.empty()) {
    for (int i = 0; i < num_joints; ++i) {
      expected_gello_joint_names_[i] =
          namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i + 1);
    }
  } else if (configured_gello_joint_names.size() != expected_gello_joint_names_.size()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "gello_joint_names has %zu entries; expected %zu",
                 configured_gello_joint_names.size(), expected_gello_joint_names_.size());
    return CallbackReturn::FAILURE;
  } else {
    std::copy(configured_gello_joint_names.begin(), configured_gello_joint_names.end(),
              expected_gello_joint_names_.begin());
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  future_timestamp_tolerance_ =
      get_node()->get_parameter("future_timestamp_tolerance").as_double();
  if (!std::isfinite(future_timestamp_tolerance_) || future_timestamp_tolerance_ < 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "future_timestamp_tolerance must be finite and non-negative (got %.6f)",
                 future_timestamp_tolerance_);
    return CallbackReturn::FAILURE;
  }
  max_gello_message_age_ = get_node()->get_parameter("max_gello_message_age").as_double();
  if (!std::isfinite(max_gello_message_age_) || max_gello_message_age_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "max_gello_message_age must be finite and positive (got %.6f)",
                 max_gello_message_age_);
    return CallbackReturn::FAILURE;
  }
  max_gello_liveness_gap_ = get_node()->get_parameter("max_gello_liveness_gap").as_double();
  if (!std::isfinite(max_gello_liveness_gap_) || max_gello_liveness_gap_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "max_gello_liveness_gap must be finite and positive (got %.6f)",
                 max_gello_liveness_gap_);
    return CallbackReturn::FAILURE;
  }
  if (max_gello_message_age_ > max_gello_liveness_gap_) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "max_gello_message_age (%.3f s) must not exceed max_gello_liveness_gap (%.3f s)",
                 max_gello_message_age_, max_gello_liveness_gap_);
    return CallbackReturn::FAILURE;
  }
  RCLCPP_INFO(get_node()->get_logger(),
              "GELLO staleness gates: max_gello_message_age = %.3f s (per-message accept), "
              "max_gello_liveness_gap = %.3f s (liveness watchdog)",
              max_gello_message_age_, max_gello_liveness_gap_);
  const auto configured_gello_joint_directions =
      get_node()->get_parameter("gello_joint_directions").as_double_array();
  if (configured_gello_joint_directions.size() != num_joints) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "gello_joint_directions has %zu entries; expected %d",
                 configured_gello_joint_directions.size(), num_joints);
    return CallbackReturn::FAILURE;
  }
  for (int i = 0; i < num_joints; ++i) {
    const double direction = configured_gello_joint_directions[i];
    if (direction != -1.0 && direction != 1.0) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "gello_joint_directions[%d] must be -1.0 or 1.0 (got %.6f)", i,
                   direction);
      return CallbackReturn::FAILURE;
    }
    gello_joint_directions_(i) = direction;
  }
  motion_generator_speed_factor_ =
      get_node()->get_parameter("motion_generator_speed_factor").as_double();
  max_goal_velocity_ = get_node()->get_parameter("max_goal_velocity").as_double();
  RCLCPP_INFO(get_node()->get_logger(),
              "approach speed_factor = %.3f, max_goal_velocity = %.3f rad/s%s",
              motion_generator_speed_factor_, max_goal_velocity_,
              max_goal_velocity_ > 0.0 ? "" : " (goal rate limiting DISABLED)");

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }

  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;

  dq_filtered_.setZero();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "gello/joint_states", 1,
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (!hasFreshGelloState_()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Refusing activation: no valid GELLO joint state received in the last %.3f s",
                 max_gello_liveness_gap_);
    return CallbackReturn::ERROR;
  }

  updateJointStates_();
  for (int i = 0; i < num_joints; ++i) {
    if (!std::isfinite(q_(i))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Refusing activation: robot joint position %d is non-finite", i);
      return CallbackReturn::ERROR;
    }
    initial_robot_position_(i) = q_(i);
    initial_gello_position_(i) = gello_position_values_[i];
  }
  mapping_references_valid_ = true;
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  q_goal_limited_valid_ = false;
  motion_generator_.reset();

  dq_filtered_.setZero();
  start_time_ = this->get_node()->now();

  RCLCPP_INFO(get_node()->get_logger(),
              "Captured GELLO/robot reference poses; mapped target starts at current robot pose");

  return CallbackReturn::SUCCESS;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  Vector7d tau_d_calculated;
  tau_d_calculated = k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  return tau_d_calculated;
}

auto JointImpedanceController::mapGelloToRobotGoal_() -> Vector7d {
  Vector7d gello_raw;
  Vector7d gello_delta;
  Vector7d q_goal;
  for (int i = 0; i < num_joints; ++i) {
    gello_raw(i) = gello_position_values_[i];
    gello_delta(i) = gello_raw(i) - initial_gello_position_(i);
    q_goal(i) =
        initial_robot_position_(i) + gello_joint_directions_(i) * gello_delta(i);
  }

  RCLCPP_DEBUG_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "GELLO map raw=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f] "
      "delta=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f] "
      "direction=[%.0f %.0f %.0f %.0f %.0f %.0f %.0f] "
      "q_goal=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
      gello_raw(0), gello_raw(1), gello_raw(2), gello_raw(3), gello_raw(4), gello_raw(5),
      gello_raw(6), gello_delta(0), gello_delta(1), gello_delta(2), gello_delta(3),
      gello_delta(4), gello_delta(5), gello_delta(6), gello_joint_directions_(0),
      gello_joint_directions_(1), gello_joint_directions_(2), gello_joint_directions_(3),
      gello_joint_directions_(4), gello_joint_directions_(5), gello_joint_directions_(6),
      q_goal(0), q_goal(1), q_goal(2), q_goal(3), q_goal(4), q_goal(5), q_goal(6));
  return q_goal;
}

bool JointImpedanceController::validateGains_(const std::vector<double>& gains,
                                              const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set", gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), num_joints, gains.size());
    return false;
  }

  return true;
}

bool JointImpedanceController::validateGelloJointState_(
    const sensor_msgs::msg::JointState& msg, const rclcpp::Time& receive_time) {
  auto logger = get_node()->get_logger();
  auto clock = get_node()->get_clock();

  if (msg.position.size() != gello_position_values_.size()) {
    RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                         "Rejecting GELLO state: unexpected position count %zu (expected %zu)",
                         msg.position.size(), gello_position_values_.size());
    return false;
  }
  if (msg.name.size() != expected_gello_joint_names_.size()) {
    RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                         "Rejecting GELLO state: unexpected joint-name count %zu (expected %zu)",
                         msg.name.size(), expected_gello_joint_names_.size());
    return false;
  }
  for (std::size_t i = 0; i < expected_gello_joint_names_.size(); ++i) {
    if (msg.name[i] != expected_gello_joint_names_[i]) {
      RCLCPP_WARN_THROTTLE(
          logger, *clock, 2000,
          "Rejecting GELLO state: joint name mismatch at index %zu: got '%s', expected '%s'",
          i, msg.name[i].c_str(), expected_gello_joint_names_[i].c_str());
      return false;
    }
    if (!std::isfinite(msg.position[i])) {
      RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                           "Rejecting GELLO state: non-finite position at index %zu (%s)", i,
                           expected_gello_joint_names_[i].c_str());
      return false;
    }
  }

  const double timestamp_age = (receive_time - rclcpp::Time(msg.header.stamp)).seconds();
  if (timestamp_age < -future_timestamp_tolerance_) {
    RCLCPP_WARN_THROTTLE(
        logger, *clock, 2000,
        "Rejecting GELLO state: timestamp too far in the future "
        "(lead %.6f s, tolerance %.6f s)",
        -timestamp_age, future_timestamp_tolerance_);
    return false;
  }
  if (timestamp_age >= max_gello_message_age_) {
    RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
                         "Rejecting GELLO state: message too old "
                         "(timestamp age %.6f s, limit %.3f s)",
                         timestamp_age, max_gello_message_age_);
    return false;
  }
  return true;
}

bool JointImpedanceController::hasFreshGelloState_() const {
  if (!gello_position_values_valid_ || last_valid_gello_receive_time_.seconds() == 0.0) {
    return false;
  }
  const double reception_age =
      (get_node()->now() - last_valid_gello_receive_time_).seconds();
  return reception_age >= 0.0 && reception_age < max_gello_liveness_gap_;
}

void JointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool JointImpedanceController::initializeMotionGenerator_() {
  if (!hasFreshGelloState_() || !mapping_references_valid_) {
    // Only send a warning once every 10 seconds in order not to spam the log
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for valid joint states...");
    return false;
  }

  updateJointStates_();
  const Vector7d q_goal = mapGelloToRobotGoal_();
  RCLCPP_INFO(get_node()->get_logger(), "q_goal of motion generator: [%f, %f, %f, %f, %f, %f, %f]",
              q_goal(0), q_goal(1), q_goal(2), q_goal(3), q_goal(4), q_goal(5), q_goal(6));

  motion_generator_ =
      std::make_unique<MotionGenerator>(motion_generator_speed_factor_, q_, q_goal);

  // Start the trajectory clock HERE, not at on_activate. The controller activates as soon
  // as it is spawned but the generator is only built once the first GELLO message arrives,
  // which can be many seconds later. Timing the trajectory from activation means
  // `trajectory_time` is already past the end on the very first evaluation, so the
  // generator returns its final point immediately and the arm is commanded to jump to the
  // GELLO pose in a single cycle - a large step into a stiff impedance law, which trips
  // joint_velocity_violation. Resetting it here makes the approach actually ramp.
  start_time_ = get_node()->now();
  return true;
}

}  // namespace franka_fr3_arm_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
