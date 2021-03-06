#include <franka_example_controllers/cartesian_impedance_example_controller.h>
#include <cmath>
#include <memory>
#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include "pseudo_inversion.h"
#include <rosbag/bag.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

namespace franka_example_controllers {

bool CartesianImpedanceExampleController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  return true;
}

void CartesianImpedanceExampleController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  franka::RobotState robot_state = state_handle_->getRobotState();

  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());
  // Bias correction for the current external torque
  tau_ext_initial_ = tau_measured - gravity;
  tau_error_.setZero();

  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());

}

void CartesianImpedanceExampleController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {

  auto Kp = 15;
  auto Kv = 2*sqrt(Kp);
  double error_mag;

  // get state variables
  franka::RobotState initial_state = state_handle_->getRobotState();
  franka::RobotState robot_state = state_handle_->getRobotState();

  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 7> gravity_array = model_handle_->getGravity();

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());

  Eigen::Map<Eigen::Matrix<double, 7, 1>> gravity(gravity_array.data());

  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  // convert to Eigen
  Eigen::MatrixXd torque_matrix;

  std::array<double, 49> mass = model_handle_->getMass();

  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> mass_joint(mass.data());
  initial_pose_ = initial_state.O_T_EE_d;

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::MatrixXd jacobian_transpose_pinv;
  Eigen::MatrixXd jacobian_inv;
  Eigen::MatrixXd mass_cartesian;
  Eigen::MatrixXd gravity_cartesian;
  Eigen::MatrixXd velocity;

  std::array<double, 6> desired_pose;
  std::array<double, 6> current_pose_array;

  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  pseudoInverse(jacobian, jacobian_inv);

  mass_cartesian = jacobian_transpose_pinv*mass_joint*jacobian_inv;
//  gravity_cartesian = jacobian_transpose_pinv*gravity_joint;
  velocity = jacobian*dq;

//Cartesian Position Regulator
  elapsed_time_ += period;
  ros::Duration t_f(10);
  double t_f_d = t_f.toSec();

  double elapsed_time_d = elapsed_time_.toSec();
  int elapsed_time_ = static_cast<int>(elapsed_time_d);

  double x_d_pose = 0.306588;
  double y_d_pose = -0.4;
  double z_d_pose = 0.486736;

  std::array<double, 16> new_pose = initial_pose_;
  std::array<double, 16> current_pose_ = state_handle_->getRobotState().O_T_EE;

  new_pose[12] = x_d_pose;
  new_pose[13] = y_d_pose;
  new_pose[14] = z_d_pose;

  for (int k = 0;  k < 3; k++){

  desired_pose[k] = new_pose[12+k];
  current_pose_array[k] = current_pose_[12+k];

  }

  desired_pose[3] = 0;
  current_pose_array[3] = 0;
  desired_pose[4] = 0;
  current_pose_array[4] = 0;
  desired_pose[5] = 0;
  current_pose_array[5] = 0;

  Eigen::Map<Eigen::Matrix<double, 6, 1>> desired_pose_vector(desired_pose.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1>> current_pose_vector(current_pose_array.data());

  Eigen::Matrix<double, 6, 1> error;
  error << desired_pose_vector - current_pose_vector;
  ROS_INFO_STREAM(error);

  error_mag = error.norm();
  ROS_INFO_STREAM(error_mag);

  Eigen::VectorXd tau_d(7), desired_force_torque(6), tau_cmd(7), tau_ext(7);
  desired_force_torque.setZero();
  desired_force_torque(2) = desired_mass_ * (3.5);
  ROS_INFO_STREAM(desired_force_torque);
  tau_ext = tau_measured - gravity - tau_ext_initial_;
  tau_d << jacobian.transpose() * desired_force_torque;
  tau_error_ = tau_error_ + period.toSec() * (tau_d - tau_ext);
  // FF + PI control (PI gains are initially all 0)
  tau_cmd = tau_d + k_p_ * (tau_d - tau_ext) + k_i_ * tau_error_;

  if (error_mag > 0.05)
  {
    torque_matrix = jacobian.transpose()*(mass_cartesian*(Kp*error - Kv*velocity));
    Eigen::Map<Eigen::Matrix<double, 7, 1>> torque_vectors(torque_matrix.data());
    tau_cmd << saturateTorqueRate(tau_cmd + torque_vectors, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_cmd(i));
    }

  }

  else
  {
    for (size_t i = 0; i < 7; ++i) {
    //  std::cout<<torque_vectors(i)<<std::endl;
      joint_handles_[i].setCommand(0.0);
    }

  }
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
controller_interface::ControllerBase)

