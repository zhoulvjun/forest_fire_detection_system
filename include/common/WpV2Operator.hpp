/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: WpV2Operator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-25
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef __WPV2OPERATOR_HPP__
#define __WPV2OPERATOR_HPP__

/* dji_osdk */
#include <dji_type.hpp>
#include <dji_mission_type.hpp>

#include <ros/ros.h>

/* dji_osdk_ros */
#include <dji_osdk_ros/DownloadWaypointV2Mission.h>
#include <dji_osdk_ros/GetGlobalCruisespeed.h>
#include <dji_osdk_ros/PauseWaypointV2Mission.h>
#include <dji_osdk_ros/ResumeWaypointV2Mission.h>
#include <dji_osdk_ros/SetGlobalCruisespeed.h>
#include <dji_osdk_ros/StartWaypointV2Mission.h>
#include <dji_osdk_ros/StopWaypointV2Mission.h>
#include <dji_osdk_ros/UploadWaypointV2Action.h>
#include <dji_osdk_ros/UploadWaypointV2Mission.h>

namespace ffds_commom {

  using namespace DJI::OSDK;

class WpV2Operator {
public:
  void setWaypointV2Defaults(dji_osdk_ros::WaypointV2 &waypointV2);
  bool uploadWaypointV2Mission(ros::NodeHandle &nh);
  bool uploadWaypointV2Action(ros::NodeHandle &nh);
  bool
  downloadWaypointV2Mission(ros::NodeHandle &nh,
                            std::vector<dji_osdk_ros::WaypointV2> &mission);
  bool startWaypointV2Mission(ros::NodeHandle &nh);
  bool stopWaypointV2Mission(ros::NodeHandle &nh);
  bool pauseWaypointV2Mission(ros::NodeHandle &nh);
  bool resumeWaypointV2Mission(ros::NodeHandle &nh);
  bool setGlobalCruiseSpeed(ros::NodeHandle &nh, float32_t cruiseSpeed);
  float32_t getGlobalCruiseSpeed(ros::NodeHandle &nh);

private:
  dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
  dji_osdk_ros::UploadWaypointV2Action uploadWaypointV2Action_;
  dji_osdk_ros::DownloadWaypointV2Mission downloadWaypointV2Mission_;
  dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
  dji_osdk_ros::StopWaypointV2Mission stopWaypointV2Mission_;
  dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
  dji_osdk_ros::ResumeWaypointV2Mission resumeWaypointV2Mission_;
  dji_osdk_ros::SetGlobalCruisespeed setGlobalCruisespeed_;
  dji_osdk_ros::GetGlobalCruisespeed getGlobalCruisespeed_;

  ros::ServiceClient waypointV2_upload_mission_client;
  ros::ServiceClient waypointV2_upload_action_client;
  ros::ServiceClient waypointV2_download_mission_client;
  ros::ServiceClient waypointV2_start_mission_client;
  ros::ServiceClient waypointV2_stop_mission_client;
  ros::ServiceClient waypointV2_pause_mission_client;
  ros::ServiceClient waypointV2_resume_mission_client;
  ros::ServiceClient waypointV2_set_global_cruisespeed_client;
  ros::ServiceClient waypointV2_get_global_cruisespeed_client;
};

} // namespace ffds_commom

#endif /* WPV2OPERATOR_HPP */
