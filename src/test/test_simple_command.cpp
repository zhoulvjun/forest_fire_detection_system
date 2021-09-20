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

#include "ros/init.h"
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

int TestSimpleCommand::run() {

  ros::Rate rate(1);
  begin_time = ros::Time::now();

  while (ros::ok()) {
    ROS_INFO("Hello");
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_simple_command_node");
  TestSimpleCommand node;
  node.run();
  return 0;
}
