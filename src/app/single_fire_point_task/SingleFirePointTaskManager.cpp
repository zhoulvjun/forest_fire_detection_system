/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: single_fire_point_task_manager.cpp
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

#include <app/single_fire_point_task/SingleFirePointTaskManager.hpp>
using namespace FFDS::APP;

sensor_msgs::NavSatFix SingleFirePointTaskManager::getHomeGPosAverage(int times) {

  sensor_msgs::NavSatFix homeGPos;

  for (int i = 0; i < times; i++) {
    ros::spinOnce();
    homeGPos.latitude += gps_position_.latitude;
    homeGPos.longitude += gps_position_.longitude;
    homeGPos.altitude += gps_position_.altitude;
  }
  homeGPos.latitude = homeGPos.latitude / times;
  homeGPos.longitude = homeGPos.longitude / times;
  homeGPos.altitude = homeGPos.altitude / times;

  return homeGPos;
}

matrix::Eulerf SingleFirePointTaskManager::getHomeHeadingAverage(int times) {

  geometry_msgs::QuaternionStamped quat;

  for (int i = 0; i < times; i++) {
    ros::spinOnce();
    quat.quaternion.w += attitude_data_.quaternion.w;
    quat.quaternion.x += attitude_data_.quaternion.x;
    quat.quaternion.y += attitude_data_.quaternion.y;
    quat.quaternion.z += attitude_data_.quaternion.z;
  }
  quat.quaternion.w = quat.quaternion.w / times;
  quat.quaternion.x = quat.quaternion.x / times;
  quat.quaternion.y = quat.quaternion.y / times;
  quat.quaternion.z = quat.quaternion.z / times;
  matrix::Quaternionf average_quat(quat.quaternion.w, quat.quaternion.x, quat.quaternion.y, quat.quaternion.z);

  return matrix::Eulerf(average_quat);
}

void SingleFirePointTaskManager::attitudeSubCallback(
    const geometry_msgs::QuaternionStampedConstPtr &attitudeData) {
  attitude_data_ = *attitudeData;
}

void SingleFirePointTaskManager::gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition) {

  gps_position_ = *gpsPosition;
}

void SingleFirePointTaskManager::run() {

  MODULES::ZigzagPathPlanner pathPlanner(getHomeGPosAverage(100), 10, 100.0, 40, 15);
  MODULES::WpV2Operator wpV2Operator(nh);

  /* 1. generate path */
}

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "single_fire_point_task_manager_node");

  SingleFirePointTaskManager taskManager;
  taskManager.run();
  return 0;
}
