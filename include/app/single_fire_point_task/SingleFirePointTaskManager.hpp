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

namespace FFDS {
namespace APP {

class SingleFirePointTaskManager {

private:
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub;
  ros::Subscriber attitudeSub;

  sensor_msgs::NavSatFix gps_position_;
  geometry_msgs::QuaternionStamped attitude_data_;

  void readPathParams();

  sensor_msgs::NavSatFix getHomeGPos(int times);

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
  };

  void run();
};

} // namespace APP
} // namespace FFDS

#endif /* SINGLEFIREPOINTTASKMANAGER_HPP */
