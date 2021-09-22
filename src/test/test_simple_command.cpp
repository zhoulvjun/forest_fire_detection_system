/*******************************************************************************
 *
 *   Copyright (C) 2021 Lee Ltd. All rights reserved.
 *
 *   @Filename: test_simple_command.cpp
 *
 *   @Author: lee-shun
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-19
 *
 *   @Description:
 *
 ******************************************************************************/

#include <test/test_simple_command.hpp>

TestSimpleCommand::TestSimpleCommand() {

  vehicle_att_subscriber = nh.subscribe<geometry_msgs::QuaternionStamped>(
      "dji_osdk_ros/attitude", 10, &TestSimpleCommand::vehical_att_cb, this);

  task_control_client =
      nh.serviceClient<dji_osdk_ros::FlightTaskControl>("/flight_task_control");

  set_joystick_mode_client =
      nh.serviceClient<dji_osdk_ros::SetJoystickMode>("set_joystick_mode");
}

TestSimpleCommand::~TestSimpleCommand() {}

void TestSimpleCommand::vehical_att_cb(
    const geometry_msgs::QuaternionStamped::ConstPtr &msg) {
  vehical_att = *msg;
}

void TestSimpleCommand::print_vehical_att(
    const geometry_msgs::QuaternionStamped &att) {
  ROS_INFO("the quaternion is:\n");
  ROS_INFO("w:%.2f\n", att.quaternion.w);
  ROS_INFO("x:%.2f\n", att.quaternion.x);
  ROS_INFO("y:%.2f\n", att.quaternion.y);
  ROS_INFO("z:%.2f\n", att.quaternion.z);
}

/**
 * @param[in]
 * @param[out]
 * @return
 * @ref
 * @see
 * @note the function to generate the zegzag rectangle line command using the
 * the dji FlightTaskControl msg.
 */
std::vector<dji_osdk_ros::JoystickCommand>
TestSimpleCommand::gernate_rectangle_command(float len, float wid, float num) {
  float each_len = len / num;
  int point_num = 2 * (num + 1);
  dji_osdk_ros::JoystickCommand command;
  std::vector<dji_osdk_ros::JoystickCommand> ctrl_vec;

  bool is_lower_left = true;
  bool is_upper_left = false;
  bool is_lower_right = false;
  bool is_upper_right = false;

  // turn left is positive
  for (int i = 0; i < point_num - 1; ++i) {

    if (is_lower_left) {
      command.x = wid;
      command.yaw = 90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = true;
      is_lower_right = false;
      is_upper_right = false;
    } else if (is_upper_left) {
      command.x = each_len;
      command.yaw = -90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = true;
    } else if (is_upper_right) {
      command.x = wid;
      command.yaw = -90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = true;
      is_upper_right = false;
    } else if (is_lower_right) {
      command.x = each_len;
      command.yaw = 90.0;
      ctrl_vec.push_back(command);

      is_lower_left = true;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = false;
    } else {
      ROS_INFO("the bool is wrong!");
    }
  }

  return ctrl_vec;
}

void TestSimpleCommand::print_control_command(
    const std::vector<dji_osdk_ros::JoystickCommand> &ctrl_command_vec) {

  for (int i = 0; i < ctrl_command_vec.size(); ++i) {
    std::cout << "point:" << i << "-------" << std::endl;
    auto em = ctrl_command_vec[i];
    std::cout << "yaw:" << em.yaw << std::endl;
    std::cout << "x:" << em.x << std::endl;
  }
}

bool TestSimpleCommand::moveByPosOffset(
    dji_osdk_ros::FlightTaskControl &task,
    const dji_osdk_ros::JoystickCommand &offsetDesired, float posThresholdInM,
    float yawThresholdInDeg) {

  task.request.task =
      dji_osdk_ros::FlightTaskControl::Request::TASK_POSITION_AND_YAW_CONTROL;
  task.request.joystickCommand.x = offsetDesired.x;
  task.request.joystickCommand.y = offsetDesired.y;
  task.request.joystickCommand.z = offsetDesired.z;
  task.request.joystickCommand.yaw = offsetDesired.yaw;
  task.request.posThresholdInM = posThresholdInM;
  task.request.yawThresholdInDeg = yawThresholdInDeg;

  task_control_client.call(task);
  return task.response.result;
}

int TestSimpleCommand::run() {

  ros::Rate rate(1);
  begin_time = ros::Time::now();
  char inputChar;
  TestSimpleCommand node;


  auto command_vec = node.gernate_rectangle_command(10.0, 3.0, 2);
  node.print_control_command(command_vec);

  dji_osdk_ros::SetJoystickMode joystickMode;

  ROS_INFO_STREAM("set the body axis!");
  joystickMode.request.yaw_mode = joystickMode.request.HORIZONTAL_BODY;
  set_joystick_mode_client.call(joystickMode);


  ROS_INFO_STREAM(
      "command generating finished, if you are ready to take off? y/n");
  std::cin >> inputChar;

  if (inputChar == 'n') {
    ROS_INFO_STREAM("exist!");
    return 0;
  } else {
    control_task.request.task =
        dji_osdk_ros::FlightTaskControl::Request::TASK_TAKEOFF;
    ROS_INFO_STREAM("Takeoff request sending ...");
    task_control_client.call(control_task);
    if (control_task.response.result == false) {
      ROS_ERROR_STREAM("Takeoff task failed");
    } else {
      ROS_INFO_STREAM("Takeoff task successful");
      ros::Duration(2.0).sleep();
      ROS_INFO_STREAM("Move by position offset request sending ...");

      for (int i = 0; ros::ok() && (i < command_vec.size()); ++i) {
        ROS_INFO_STREAM("moving to" << i << "point");
        moveByPosOffset(control_task, command_vec[i], 0.8, 1);
      }

      control_task.request.task =
          dji_osdk_ros::FlightTaskControl::Request::TASK_LAND;
      ROS_INFO_STREAM("Landing request sending ...");
      task_control_client.call(control_task);
      if (control_task.response.result == true) {
        ROS_INFO_STREAM("Land task successful");
      } else {
        ROS_INFO_STREAM("Land task failed.");
      }
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_simple_command_node");
  TestSimpleCommand node;
  node.run();
  return 0;
}
