#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace
{
std::vector<double> parameter_vector_or_default(
  rclcpp::Node & node, const std::string & name, const std::vector<double> & fallback)
{
  auto value = node.declare_parameter<std::vector<double>>(name, fallback);
  return value.size() == fallback.size() ? value : fallback;
}
}  // namespace

class SimEffortEstimator : public rclcpp::Node
{
public:
  SimEffortEstimator() : Node("sim_effort_estimator")
  {
    input_joint_states_topic_ = declare_parameter<std::string>(
      "input_joint_states_topic", "/joint_state_broadcaster/joint_states");
    output_joint_states_topic_ = declare_parameter<std::string>(
      "output_joint_states_topic", "/joint_states");
    controller_state_topic_ = declare_parameter<std::string>(
      "controller_state_topic", "/jaka_s12_controller/controller_state");

    const std::vector<double> default_p = {100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
    const std::vector<double> default_i = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    const std::vector<double> default_d = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    const std::vector<double> default_i_clamp = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    p_gains_ = parameter_vector_or_default(*this, "p_gains", default_p);
    i_gains_ = parameter_vector_or_default(*this, "i_gains", default_i);
    d_gains_ = parameter_vector_or_default(*this, "d_gains", default_d);
    i_clamps_ = parameter_vector_or_default(*this, "i_clamps", default_i_clamp);

    effort_deadband_ = declare_parameter<double>("effort_deadband", 0.05);
    max_effort_ = declare_parameter<double>("max_effort", 500.0);
    smoothing_alpha_ = std::clamp(declare_parameter<double>("smoothing_alpha", 0.35), 0.0, 1.0);

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(output_joint_states_topic_, 10);

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      input_joint_states_topic_, 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        publish_with_effort(*msg);
      });

    controller_state_sub_ = create_subscription<control_msgs::msg::JointTrajectoryControllerState>(
      controller_state_topic_, 10,
      [this](const control_msgs::msg::JointTrajectoryControllerState::SharedPtr msg) {
        update_effort_estimate(*msg);
      });

    RCLCPP_INFO(
      get_logger(), "Estimating simulated joint effort from %s and publishing %s",
      controller_state_topic_.c_str(), output_joint_states_topic_.c_str());
  }

private:
  void update_effort_estimate(const control_msgs::msg::JointTrajectoryControllerState & state)
  {
    const rclcpp::Time stamp(state.header.stamp);
    double dt = 0.0;
    if (have_last_state_stamp_ && stamp > last_state_stamp_) {
      dt = (stamp - last_state_stamp_).seconds();
    }
    last_state_stamp_ = stamp;
    have_last_state_stamp_ = true;

    for (size_t i = 0; i < state.joint_names.size(); ++i) {
      const std::string & name = state.joint_names[i];
      double effort = output_effort_if_available(state, i);

      if (!std::isfinite(effort)) {
        const double position_error = i < state.error.positions.size() ? state.error.positions[i] : 0.0;
        const double velocity_error = i < state.error.velocities.size() ? state.error.velocities[i] : 0.0;
        double & integral = integral_error_by_joint_[name];

        if (dt > 0.0 && dt < 0.5) {
          integral += position_error * dt;
          integral = std::clamp(integral, -i_clamps_[i], i_clamps_[i]);
        }

        effort =
          p_gains_[i] * position_error +
          i_gains_[i] * integral +
          d_gains_[i] * velocity_error;
      }

      if (std::abs(effort) < effort_deadband_) {
        effort = 0.0;
      }
      effort = std::clamp(effort, -max_effort_, max_effort_);

      auto previous_it = effort_by_joint_.find(name);
      if (previous_it != effort_by_joint_.end()) {
        effort = smoothing_alpha_ * effort + (1.0 - smoothing_alpha_) * previous_it->second;
      }
      effort_by_joint_[name] = effort;
    }
  }

  double output_effort_if_available(
    const control_msgs::msg::JointTrajectoryControllerState & state, size_t index) const
  {
    if (index >= state.output.effort.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double effort = state.output.effort[index];
    if (!std::isfinite(effort) || std::abs(effort) < 1e-9) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return effort;
  }

  void publish_with_effort(const sensor_msgs::msg::JointState & input)
  {
    auto output = input;
    output.effort.assign(output.name.size(), 0.0);

    for (size_t i = 0; i < output.name.size(); ++i) {
      const auto effort_it = effort_by_joint_.find(output.name[i]);
      if (effort_it != effort_by_joint_.end()) {
        output.effort[i] = effort_it->second;
      }
    }

    joint_state_pub_->publish(output);
  }

  std::string input_joint_states_topic_;
  std::string output_joint_states_topic_;
  std::string controller_state_topic_;
  std::vector<double> p_gains_;
  std::vector<double> i_gains_;
  std::vector<double> d_gains_;
  std::vector<double> i_clamps_;
  double effort_deadband_;
  double max_effort_;
  double smoothing_alpha_;
  rclcpp::Time last_state_stamp_;
  bool have_last_state_stamp_ = false;

  std::unordered_map<std::string, double> integral_error_by_joint_;
  std::unordered_map<std::string, double> effort_by_joint_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<control_msgs::msg::JointTrajectoryControllerState>::SharedPtr controller_state_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimEffortEstimator>());
  rclcpp::shutdown();
  return 0;
}
