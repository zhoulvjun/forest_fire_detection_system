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
std::vector<TestSimpleCommand::ControlCommand>
TestSimpleCommand::gernate_rectangle_command(float len, float wid, float num) {
  float each_len = len / num;
  int point_num = 2 * (num + 1);
  ControlCommand command(0.0, 0.0, 0.0, 0.0);
  std::vector<TestSimpleCommand::ControlCommand> ctrl_vec;

  bool is_lower_left = true;
  bool is_upper_left = false;
  bool is_lower_right = false;
  bool is_upper_right = false;

  // turn left is positive
  for (int i = 0; i < point_num-1; ++i) {

    if (is_lower_left) {
      command.offset_x = wid;
      command.offset_yaw = 90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = true;
      is_lower_right = false;
      is_upper_right = false;
    } else if (is_upper_left) {
      command.offset_x = each_len;
      command.offset_yaw = -90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = true;
    } else if (is_upper_right) {
      command.offset_x = wid;
      command.offset_yaw = -90.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = true;
      is_upper_right = false;
    } else if (is_lower_right) {
      command.offset_x = each_len;
      command.offset_yaw = 90.0;
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

int TestSimpleCommand::run() {

  ros::Rate rate(1);
  begin_time = ros::Time::now();

  while (ros::ok()) {
    ROS_DEBUG("hello test simple command!");
    print_vehical_att(vehical_att);
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_simple_command_node");
  TestSimpleCommand node;

  auto test_vec = node.gernate_rectangle_command(10.0, 3.0, 2);

  for (int i = 0; i < test_vec.size(); ++i) {
    std::cout << "point:" << i <<"-------"<< std::endl;
    auto em = test_vec[i];
    std::cout << "yaw:" << em.offset_yaw << std::endl;
    std::cout << "x:" << em.offset_x << std::endl;
  }
  return 0;
}
