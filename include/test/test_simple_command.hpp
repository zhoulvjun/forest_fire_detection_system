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

  geometry_msgs::QuaternionStamped vehical_att;

  /**
   * the callback functions
   * */
  void vehical_att_cb(const geometry_msgs::QuaternionStamped::ConstPtr &msg);

  /**
   * functions
   * */

  void print_vehical_att(const geometry_msgs::QuaternionStamped &att);

public:
  struct ControlCommand {
    float offset_x;
    float offset_y;
    float offset_z;
    float offset_yaw;

    ControlCommand(float x, float y, float z, float yaw)
        : offset_x(x), offset_y(y), offset_z(z), offset_yaw(yaw){};
  };
  TestSimpleCommand();
  ~TestSimpleCommand();

  std::vector<TestSimpleCommand::ControlCommand>
  gernate_rectangle_command(float len, float wid, float num);

  int run();
};

#endif /* TEST_SIMPLE_COMMAND_HPP */
