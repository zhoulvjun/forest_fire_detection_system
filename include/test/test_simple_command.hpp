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
 *   @Description: use the joystick srv, simple command.
 *
 ******************************************************************************/

#ifndef __TEST_SIMPLE_COMMAND_HPP__
#define __TEST_SIMPLE_COMMAND_HPP__

// dji
#include <dji_osdk_ros/FlightTaskControl.h>
#include <dji_osdk_ros/JoystickAction.h>
#include <dji_osdk_ros/common_type.h>

// ros
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/ros.h>
#include <ros/time.h>

// c++
#include <iostream>
#include <vector>

class TestSimpleCommand {

private:
  ros::NodeHandle nh;
  ros::Time begin_time;

  ros::Subscriber vehicle_att_subscriber;
  ros::ServiceClient task_control_client;

  geometry_msgs::QuaternionStamped vehical_att;
  dji_osdk_ros::FlightTaskControl control_task;

  /**
   * the callback functions
   * */
  void vehical_att_cb(const geometry_msgs::QuaternionStamped::ConstPtr &msg);

  /**
   * functions
   * */
  void print_vehical_att(const geometry_msgs::QuaternionStamped &att);

public:

  TestSimpleCommand();
  ~TestSimpleCommand();

  int run();

  std::vector<dji_osdk_ros::JoystickCommand>
  gernate_rectangle_command(float len, float wid, float num);

  bool moveByPosOffset(dji_osdk_ros::FlightTaskControl &task,
                       const dji_osdk_ros::JoystickCommand &offsetDesired,
                       float posThresholdInM, float yawThresholdInDeg);
  void
  print_control_command(const std::vector<dji_osdk_ros::JoystickCommand> &ctrl_command_vec);
};

#endif /* TEST_SIMPLE_COMMAND_HPP */
