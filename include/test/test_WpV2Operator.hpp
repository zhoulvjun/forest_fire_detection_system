/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
 *
 *   @Filename: test_WpV2Operator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-19
 *
 *   @Description: Rewrite of the waypointv2 node example
 *
 ******************************************************************************/

#ifndef __TEST_WPV2OPERATOR_HPP__
#define __TEST_WPV2OPERATOR_HPP__

/* debug */
#include <dbg-macro/dbg.h>

/* ros */
#include <ros/ros.h>

/* mesages */
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <sensor_msgs/NavSatFix.h>

#include <common/WpV2Operator.hpp>
#include <tools/MathLib.hpp>

/* dji_osdk_ros */
#include <dji_osdk_ros/common_type.h>
#include <dji_osdk_ros/GetDroneType.h>
#include <dji_osdk_ros/InitWaypointV2Setting.h>
#include <dji_osdk_ros/GenerateWaypointV2Action.h>
#include <dji_osdk_ros/SubscribeWaypointV2Event.h>
#include <dji_osdk_ros/SubscribeWaypointV2State.h>
#include <dji_osdk_ros/WaypointV2MissionEventPush.h>
#include <dji_osdk_ros/WaypointV2MissionStatePush.h>

/**
 * global variable
 * */
dji_osdk_ros::GetDroneType drone_type;
dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
dji_osdk_ros::UploadWaypointV2Action uploadWaypointV2Action_;
dji_osdk_ros::DownloadWaypointV2Mission downloadWaypointV2Mission_;
dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
dji_osdk_ros::StopWaypointV2Mission stopWaypointV2Mission_;
dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
dji_osdk_ros::ResumeWaypointV2Mission resumeWaypointV2Mission_;
dji_osdk_ros::SetGlobalCruisespeed setGlobalCruisespeed_;
dji_osdk_ros::GetGlobalCruisespeed getGlobalCruisespeed_;
dji_osdk_ros::GenerateWaypointV2Action generateWaypointV2Action_;
dji_osdk_ros::SubscribeWaypointV2Event subscribeWaypointV2Event_;
dji_osdk_ros::SubscribeWaypointV2State subscribeWaypointV2State_;

ros::ServiceClient waypointV2_init_setting_client;
ros::ServiceClient waypointV2_upload_mission_client;
ros::ServiceClient waypointV2_upload_action_client;
ros::ServiceClient waypointV2_download_mission_client;
ros::ServiceClient waypointV2_start_mission_client;
ros::ServiceClient waypointV2_stop_mission_client;
ros::ServiceClient waypointV2_pause_mission_client;
ros::ServiceClient waypointV2_resume_mission_client;
ros::ServiceClient waypointV2_set_global_cruisespeed_client;
ros::ServiceClient waypointV2_get_global_cruisespeed_client;
ros::ServiceClient waypointV2_generate_actions_client;
ros::ServiceClient waypointV2_mission_event_push_client;
ros::ServiceClient waypointV2_mission_state_push_client;

ros::Subscriber waypointV2EventSub;
ros::Subscriber waypointV2StateSub;

ros::ServiceClient get_drone_type_client;
sensor_msgs::NavSatFix gps_position_;
dji_osdk_ros::WaypointV2MissionEventPush waypoint_V2_mission_event_push_;
dji_osdk_ros::WaypointV2MissionStatePush waypoint_V2_mission_state_push_;

void gpsPositionSubCallback(const sensor_msgs::NavSatFix::ConstPtr& gpsPosition);
void waypointV2MissionStateSubCallback(const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr& waypointV2MissionStatePush);
void waypointV2MissionEventSubCallback(const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr& waypointV2MissionEventPush);

void setWaypointV2Defaults(dji_osdk_ros::WaypointV2& waypointV2);
std::vector<dji_osdk_ros::WaypointV2> generatePolygonWaypoints(const ros::NodeHandle &nh, DJI::OSDK::float32_t radius, uint16_t polygonNum);
bool initWaypointV2Setting(ros::NodeHandle &nh);
bool uploadWaypointV2Mission(ros::NodeHandle &nh);
bool uploadWaypointV2Action(ros::NodeHandle &nh);
bool downloadWaypointV2Mission(ros::NodeHandle &nh, std::vector<dji_osdk_ros::WaypointV2> &mission);
bool startWaypointV2Mission(ros::NodeHandle &nh);
bool stopWaypointV2Mission(ros::NodeHandle &nh);
bool pauseWaypointV2Mission(ros::NodeHandle &nh);
bool resumeWaypointV2Mission(ros::NodeHandle &nh);
bool generateWaypointV2Actions(ros::NodeHandle &nh, uint16_t actionNum);
bool setGlobalCruiseSpeed(ros::NodeHandle &nh, DJI::OSDK::float32_t cruiseSpeed);
DJI::OSDK::float32_t getGlobalCruiseSpeed(ros::NodeHandle &nh);

bool runWaypointV2Mission(ros::NodeHandle &nh);


#endif
