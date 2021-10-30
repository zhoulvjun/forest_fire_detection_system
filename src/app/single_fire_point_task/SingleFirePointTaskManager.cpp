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

matrix::Eulerf SingleFirePointTaskManager::getInitAttAverage(int times) {

  geometry_msgs::QuaternionStamped quat;

  for (int i = 0; i < times; i++) {
    quat.quaternion.w += attitude_data_.quaternion.w;
    quat.quaternion.x += attitude_data_.quaternion.x;
    quat.quaternion.y += attitude_data_.quaternion.y;
    quat.quaternion.z += attitude_data_.quaternion.z;
    ros::spinOnce();
  }
  quat.quaternion.w = quat.quaternion.w / times;
  quat.quaternion.x = quat.quaternion.x / times;
  quat.quaternion.y = quat.quaternion.y / times;
  quat.quaternion.z = quat.quaternion.z / times;

  ROS_INFO_STREAM(quat.quaternion.w);
  matrix::Quaternionf average_quat(quat.quaternion.w, quat.quaternion.x, quat.quaternion.y, quat.quaternion.z);

  return matrix::Eulerf(average_quat);
}

void SingleFirePointTaskManager::attitudeSubCallback(
    const geometry_msgs::QuaternionStampedConstPtr &attitudeData) {
  ROS_INFO_STREAM("att callback");
  attitude_data_ = *attitudeData;
}

void SingleFirePointTaskManager::gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition) {

  gps_position_ = *gpsPosition;
}

void SingleFirePointTaskManager::run() {

  sensor_msgs::NavSatFix homeGPos = getHomeGPosAverage(100);
  matrix::Eulerf initAtt = getInitAttAverage(100);

  ROS_INFO_STREAM(initAtt.psi());

  MODULES::ZigzagPathPlanner pathPlanner(homeGPos, 10, 100.0, 40, 15);
  MODULES::WpV2Operator wpV2Operator(nh);

  /* Step: 1 init the mission */
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initWaypointV2Setting_.request.waypointV2InitSettings.repeatTimes = 1;
  initWaypointV2Setting_.request.waypointV2InitSettings.finishedAction = initWaypointV2Setting_.request.waypointV2InitSettings.DJIWaypointV2MissionFinishedGoHome;
  initWaypointV2Setting_.request.waypointV2InitSettings.maxFlightSpeed = 10;
  initWaypointV2Setting_.request.waypointV2InitSettings.autoFlightSpeed = 2;
  initWaypointV2Setting_.request.waypointV2InitSettings.exitMissionOnRCSignalLost = 1;
  initWaypointV2Setting_.request.waypointV2InitSettings.gotoFirstWaypointMode = initWaypointV2Setting_.request.waypointV2InitSettings.DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;
  initWaypointV2Setting_.request.waypointV2InitSettings.mission = pathPlanner.getWpV2Vec(true, true, initAtt.psi());
  initWaypointV2Setting_.request.waypointV2InitSettings.missTotalLen = initWaypointV2Setting_.request.waypointV2InitSettings.mission.size();

}

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "single_fire_point_task_manager_node");

  SingleFirePointTaskManager taskManager;
  taskManager.run();
  return 0;
}
