/*******************************************************************************
 *
 *   Copyright (C) 2021 Lee Ltd. All rights reserved.
 *
 *   @Filename: test_simple_command.hpp
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

#ifndef __TEST_SIMPLE_COMMAND_HPP__
#define __TEST_SIMPLE_COMMAND_HPP__

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/ros.h>
#include <ros/time.h>

#include <iostream>
class TestSimpleCommand {

private:
  ros::NodeHandle nh;

  ros::Time begin_time;

  ros::Subscriber vehicle_att_subscriber;

  geometry_msgs::QuaternionStamped vehical_att;

  void vehical_att_cb(const geometry_msgs::QuaternionStamped::ConstPtr &msg);
  void print_vehical_att(const geometry_msgs::QuaternionStamped &att);

public:
  TestSimpleCommand();
  ~TestSimpleCommand();

  int run();
};

#endif /* TEST_SIMPLE_COMMAND_HPP */
