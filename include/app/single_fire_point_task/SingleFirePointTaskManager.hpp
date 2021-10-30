/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SingleFirePointTaskManager.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-22
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef __SINGLEFIREPOINTTASKMANAGER_HPP__
#define __SINGLEFIREPOINTTASKMANAGER_HPP__

#include <geometry_msgs/QuaternionStamped.h>
#include <modules/PathPlanner/ZigzagPathPlanner.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <PX4-Matrix/matrix/Euler.hpp>

namespace FFDS {
namespace APP {

class SingleFirePointTaskManager {

private:
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub;
  ros::Subscriber attitudeSub;

  ros::ServiceClient obtain_ctrl_authority_client;

  sensor_msgs::NavSatFix gps_position_;
  geometry_msgs::QuaternionStamped attitude_data_;

  void readPathParams();

  sensor_msgs::NavSatFix getHomeGPosAverage(int times);
  matrix::Eulerf getInitAttAverage(int times);

  void
  gpsPositionSubCallback(const sensor_msgs::NavSatFix::ConstPtr &gpsPosition);

  void attitudeSubCallback(
      const geometry_msgs::QuaternionStampedConstPtr &attitudeData);

public:
  SingleFirePointTaskManager() {

    gpsPositionSub =
        nh.subscribe("dji_osdk_ros/gps_position", 10,
                     &SingleFirePointTaskManager::gpsPositionSubCallback, this);

    attitudeSub =
        nh.subscribe("dji_osdk_ros/attitude", 10,
                     &SingleFirePointTaskManager::attitudeSubCallback, this);
    ros::Duration(3.0).sleep();

    ROS_INFO_STREAM("initializing Done");
  };

  void run();
};

} // namespace APP
} // namespace FFDS

#endif /* SINGLEFIREPOINTTASKMANAGER_HPP */
