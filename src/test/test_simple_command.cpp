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

void TestSimpleCommand::print_vehical_att(const geometry_msgs::QuaternionStamped &att){
    ROS_INFO("the quaternion is:\n");
    ROS_INFO("w:%.2f\n", att.quaternion.w);
    ROS_INFO("x:%.2f\n", att.quaternion.x);
    ROS_INFO("y:%.2f\n", att.quaternion.y);
    ROS_INFO("z:%.2f\n", att.quaternion.z);
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
  node.run();
  return 0;
}
