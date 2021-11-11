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

#ifndef INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_
#define INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_

#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <dji_osdk_ros/SubscribeWaypointV2Event.h>
#include <dji_osdk_ros/SubscribeWaypointV2State.h>
#include <dji_osdk_ros/WaypointV2MissionEventPush.h>
#include <dji_osdk_ros/WaypointV2MissionStatePush.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <PX4-Matrix/matrix/Euler.hpp>
#include <modules/PathPlanner/ZigzagPathPlanner.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <tools/SystemLib.hpp>

namespace FFDS {
namespace APP {

class SingleFirePointTaskManager {
 private:
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub;
  ros::Subscriber attitudeSub;
  ros::Subscriber waypointV2EventSub;
  ros::Subscriber waypointV2StateSub;

  ros::ServiceClient obtain_ctrl_authority_client;
  ros::ServiceClient waypointV2_mission_state_push_client;
  ros::ServiceClient waypointV2_mission_event_push_client;

  /**
   * ros msg
   * */

  sensor_msgs::NavSatFix gps_position_;
  geometry_msgs::QuaternionStamped attitude_data_;
  dji_osdk_ros::WaypointV2MissionEventPush waypoint_V2_mission_event_push_;
  dji_osdk_ros::WaypointV2MissionStatePush waypoint_V2_mission_state_push_;

  /**
   * ros srv
   * */

  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  dji_osdk_ros::SubscribeWaypointV2Event subscribeWaypointV2Event_;
  dji_osdk_ros::SubscribeWaypointV2State subscribeWaypointV2State_;

  /**
   * member functions
   * */

  void readPathParams();
  sensor_msgs::NavSatFix getHomeGPosAverage(int times);
  matrix::Eulerf getInitAttAverage(int times);
  void initMission(
      dji_osdk_ros::InitWaypointV2Setting *initWaypointV2SettingPtr);

  /**
   * callback functions
   * */

  void gpsPositionSubCallback(
      const sensor_msgs::NavSatFix::ConstPtr &gpsPosition);

  void attitudeSubCallback(
      const geometry_msgs::QuaternionStampedConstPtr &attitudeData);

  void waypointV2MissionEventSubCallback(
      const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
          &waypointV2MissionEventPush);

  void waypointV2MissionStateSubCallback(
      const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
          &waypointV2MissionStatePush);

 public:
  SingleFirePointTaskManager() {
    obtain_ctrl_authority_client =
        nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
            "obtain_release_control_authority");
    waypointV2_mission_state_push_client =
        nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2Event>(
            "dji_osdk_ros/waypointV2_subscribeMissionState");
    waypointV2_mission_event_push_client =
        nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2State>(
            "dji_osdk_ros/waypointV2_subscribeMissionEvent");

    gpsPositionSub =
        nh.subscribe("dji_osdk_ros/gps_position", 10,
                     &SingleFirePointTaskManager::gpsPositionSubCallback, this);
    attitudeSub =
        nh.subscribe("dji_osdk_ros/attitude", 10,
                     &SingleFirePointTaskManager::attitudeSubCallback, this);
    waypointV2EventSub = nh.subscribe(
        "dji_osdk_ros/waypointV2_mission_event", 10,
        &SingleFirePointTaskManager::waypointV2MissionEventSubCallback, this);
    waypointV2StateSub = nh.subscribe(
        "dji_osdk_ros/waypointV2_mission_state", 10,
        &SingleFirePointTaskManager::waypointV2MissionStateSubCallback, this);

    /* obtain the authorization when really needed... Now :) */
    obtainCtrlAuthority.request.enable_obtain = true;
    obtain_ctrl_authority_client.call(obtainCtrlAuthority);
    if (obtainCtrlAuthority.response.result) {
      PRINT_INFO("get control authority!");
    } else {
      PRINT_ERROR("can NOT get control authority!");
      return;
    }

    /* get the WpV2Mission states to be published ... */
    subscribeWaypointV2Event_.request.enable_sub = true;
    subscribeWaypointV2State_.request.enable_sub = true;
    waypointV2_mission_state_push_client.call(subscribeWaypointV2State_);
    waypointV2_mission_event_push_client.call(subscribeWaypointV2Event_);
    if (subscribeWaypointV2State_.response.result) {
      PRINT_INFO("get WpV2Mission state published!");
    } else {
      PRINT_ERROR("can NOT get WpV2Mission state published!");
      return;
    }
    if (subscribeWaypointV2Event_.response.result) {
      PRINT_INFO("get WpV2Mission event published!");
    } else {
      PRINT_ERROR("can NOT get WpV2Mission event published!");
      return;
    }

    /* open a thread to call the states in case of the long wait... */
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Duration(3.0).sleep();
    PRINT_INFO("initializing Done");
  }

  ~SingleFirePointTaskManager() {
    obtainCtrlAuthority.request.enable_obtain = false;
    obtain_ctrl_authority_client.call(obtainCtrlAuthority);
    if (obtainCtrlAuthority.response.result) {
      PRINT_INFO("release control authority!");
    } else {
      PRINT_ERROR("can NOT release control authority!");
    }
  }

  void run();
};

}  // namespace APP
}  // namespace FFDS

#endif  // INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_
