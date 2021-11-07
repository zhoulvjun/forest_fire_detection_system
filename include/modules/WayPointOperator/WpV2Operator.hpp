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

#include <dji_osdk_ros/DownloadWaypointV2Mission.h>
#include <dji_osdk_ros/GenerateWaypointV2Action.h>
#include <dji_osdk_ros/GetGlobalCruisespeed.h>
#include <dji_osdk_ros/InitWaypointV2Setting.h>
#include <dji_osdk_ros/PauseWaypointV2Mission.h>
#include <dji_osdk_ros/ResumeWaypointV2Mission.h>
#include <dji_osdk_ros/SetGlobalCruisespeed.h>
#include <dji_osdk_ros/StartWaypointV2Mission.h>
#include <dji_osdk_ros/StopWaypointV2Mission.h>
#include <dji_osdk_ros/UploadWaypointV2Action.h>
#include <dji_osdk_ros/UploadWaypointV2Mission.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <dji_mission_type.hpp>
#include <dji_type.hpp>
#include <vector>

namespace FFDS {
namespace MODULES {

class WpV2Operator {
  /**
   * NOTE: when calling the operators, prepare the "content" you want to pass
   * NOTE: first.
   **/

 public:
  explicit WpV2Operator(ros::NodeHandle &handle) : nh(handle) {}

  static void setWaypointV2Defaults(dji_osdk_ros::WaypointV2 &waypointV2);

  bool initWaypointV2Setting(
      dji_osdk_ros::InitWaypointV2Setting &initWaypointV2Setting_);

  bool generateWaypointV2Actions(
      dji_osdk_ros::GenerateWaypointV2Action &generateWaypointV2Action_,
      uint16_t actionNum);

  bool uploadWaypointV2Mission(
      dji_osdk_ros::UploadWaypointV2Mission &uploadWaypointV2Mission_);

  bool uploadWaypointV2Action(
      dji_osdk_ros::UploadWaypointV2Action &uploadWaypointV2Action_);

  bool downloadWaypointV2Mission(
      dji_osdk_ros::DownloadWaypointV2Mission &downloadWaypointV2Mission_,
      std::vector<dji_osdk_ros::WaypointV2> &mission);

  bool startWaypointV2Mission(
      dji_osdk_ros::StartWaypointV2Mission &startWaypointV2Mission_);

  bool stopWaypointV2Mission(
      dji_osdk_ros::StopWaypointV2Mission &stopWaypointV2Mission_);

  bool pauseWaypointV2Mission(
      dji_osdk_ros::PauseWaypointV2Mission &pauseWaypointV2Mission_);

  bool resumeWaypointV2Mission(
      dji_osdk_ros::ResumeWaypointV2Mission &resumeWaypointV2Mission_);

  bool setGlobalCruiseSpeed(
      dji_osdk_ros::SetGlobalCruisespeed &setGlobalCruisespeed_);

  DJI::OSDK::float32_t getGlobalCruiseSpeed(
      dji_osdk_ros::GetGlobalCruisespeed &getGlobalCruisespeed_);

 private:
  ros::NodeHandle &nh;

  ros::ServiceClient waypointV2_init_setting_client;
  ros::ServiceClient waypointV2_generate_actions_client;
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

} /* namespace MODULES */

} /* namespace FFDS */

#endif  // INCLUDE_MODULES_WAYPOINTOPERATOR_WPV2OPERATOR_HPP_
