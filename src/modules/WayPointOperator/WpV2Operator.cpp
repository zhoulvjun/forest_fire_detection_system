/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: WpV2Operator.cpp
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

#include <modules/WayPointOperator/WpV2Operator.hpp>

/* TODO: to deal with the local_pos and gpos, as well as the test */
void FFDS::MODULES::WpV2Operator::setWaypointV2Defaults(
    dji_osdk_ros::WaypointV2 &waypointV2) {
  waypointV2.waypointType =
      DJI::OSDK::DJIWaypointV2FlightPathModeGoToPointInAStraightLineAndStop;
  waypointV2.headingMode = DJI::OSDK::DJIWaypointV2HeadingModeAuto;
  waypointV2.config.useLocalCruiseVel = 0;
  waypointV2.config.useLocalMaxVel = 0;

  waypointV2.dampingDistance = 20;
  waypointV2.heading = 0;
  waypointV2.turnMode = DJI::OSDK::DJIWaypointV2TurnModeClockwise;

  /* waypointV2.latitude = 0; */
  /* waypointV2.longitude = 0; */
  /* waypointV2.relativeHeight = 0; */
  waypointV2.positionX = 0;
  waypointV2.positionY = 0;
  waypointV2.positionZ = 0;
  waypointV2.maxFlightSpeed = 9;
  waypointV2.autoFlightSpeed = 2;
}

bool FFDS::MODULES::WpV2Operator::initWaypointV2Setting(
    dji_osdk_ros::InitWaypointV2Setting &initWaypointV2Setting_) {
  waypointV2_init_setting_client =
      nh.serviceClient<dji_osdk_ros::InitWaypointV2Setting>(
          "dji_osdk_ros/waypointV2_initSetting");
  waypointV2_init_setting_client.call(initWaypointV2Setting_);

  if (initWaypointV2Setting_.response.result) {
    PRINT_INFO("Init mission setting successfully!\n");
  } else {
    PRINT_ERROR("Init mission setting failed!\n");
  }

  return initWaypointV2Setting_.response.result;
}

bool FFDS::MODULES::WpV2Operator::generateWaypointV2Actions(
    dji_osdk_ros::GenerateWaypointV2Action &generateWaypointV2Action_,
    uint16_t actionNum) {
  waypointV2_generate_actions_client =
      nh.serviceClient<dji_osdk_ros::GenerateWaypointV2Action>(
          "dji_osdk_ros/waypointV2_generateActions");

  waypointV2_generate_actions_client.call(generateWaypointV2Action_);

  return generateWaypointV2Action_.response.result;
}

bool FFDS::MODULES::WpV2Operator::uploadWaypointV2Mission(
    dji_osdk_ros::UploadWaypointV2Mission &uploadWaypointV2Mission_) {
  waypointV2_upload_mission_client =
      nh.serviceClient<dji_osdk_ros::UploadWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_uploadMission");

  waypointV2_upload_mission_client.call(uploadWaypointV2Mission_);

  if (uploadWaypointV2Mission_.response.result) {
    PRINT_INFO("Upload waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Upload waypoint v2 mission failed!\n");
  }

  return uploadWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::uploadWaypointV2Action(
    dji_osdk_ros::UploadWaypointV2Action &uploadWaypointV2Action_) {
  waypointV2_upload_action_client =
      nh.serviceClient<dji_osdk_ros::UploadWaypointV2Action>(
          "dji_osdk_ros/waypointV2_uploadAction");

  waypointV2_upload_action_client.call(uploadWaypointV2Action_);

  if (uploadWaypointV2Action_.response.result) {
    PRINT_INFO("Upload waypoint v2 actions successfully!\n");
  } else {
    PRINT_ERROR("Upload waypoint v2 actions failed!\n");
  }

  return uploadWaypointV2Action_.response.result;
}

bool FFDS::MODULES::WpV2Operator::downloadWaypointV2Mission(
    dji_osdk_ros::DownloadWaypointV2Mission &downloadWaypointV2Mission_,
    std::vector<dji_osdk_ros::WaypointV2> &mission) {
  waypointV2_download_mission_client =
      nh.serviceClient<dji_osdk_ros::DownloadWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_downloadMission");

  waypointV2_download_mission_client.call(downloadWaypointV2Mission_);

  mission = downloadWaypointV2Mission_.response.mission;

  if (downloadWaypointV2Mission_.response.result) {
    PRINT_INFO("Download waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Download waypoint v2 mission failed!\n");
  }

  return downloadWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::startWaypointV2Mission(
    dji_osdk_ros::StartWaypointV2Mission &startWaypointV2Mission_) {
  waypointV2_start_mission_client =
      nh.serviceClient<dji_osdk_ros::StartWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_startMission");

  waypointV2_start_mission_client.call(startWaypointV2Mission_);

  if (startWaypointV2Mission_.response.result) {
    PRINT_INFO("Start waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Start waypoint v2 mission failed!\n");
  }

  return startWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::stopWaypointV2Mission(
    dji_osdk_ros::StopWaypointV2Mission &stopWaypointV2Mission_) {
  waypointV2_stop_mission_client =
      nh.serviceClient<dji_osdk_ros::StopWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_stopMission");

  waypointV2_stop_mission_client.call(stopWaypointV2Mission_);

  if (stopWaypointV2Mission_.response.result) {
    PRINT_INFO("Stop waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Stop waypoint v2 mission failed!\n");
  }

  return stopWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::pauseWaypointV2Mission(
    dji_osdk_ros::PauseWaypointV2Mission &pauseWaypointV2Mission_) {
  waypointV2_pause_mission_client =
      nh.serviceClient<dji_osdk_ros::PauseWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_pauseMission");

  waypointV2_pause_mission_client.call(pauseWaypointV2Mission_);

  if (pauseWaypointV2Mission_.response.result) {
    PRINT_INFO("Pause waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Pause waypoint v2 mission failed!\n");
  }

  return pauseWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::resumeWaypointV2Mission(
    dji_osdk_ros::ResumeWaypointV2Mission &resumeWaypointV2Mission_) {
  waypointV2_resume_mission_client =
      nh.serviceClient<dji_osdk_ros::ResumeWaypointV2Mission>(
          "dji_osdk_ros/waypointV2_resumeMission");

  waypointV2_resume_mission_client.call(resumeWaypointV2Mission_);

  if (resumeWaypointV2Mission_.response.result) {
    PRINT_INFO("Resume Waypoint v2 mission successfully!\n");
  } else {
    PRINT_ERROR("Resume Waypoint v2 mission failed!\n");
  }

  return resumeWaypointV2Mission_.response.result;
}

bool FFDS::MODULES::WpV2Operator::setGlobalCruiseSpeed(
    dji_osdk_ros::SetGlobalCruisespeed &setGlobalCruisespeed_) {
  waypointV2_set_global_cruisespeed_client =
      nh.serviceClient<dji_osdk_ros::SetGlobalCruisespeed>(
          "dji_osdk_ros/waypointV2_setGlobalCruisespeed");

  waypointV2_set_global_cruisespeed_client.call(setGlobalCruisespeed_);

  if (setGlobalCruisespeed_.response.result) {
    PRINT_INFO("Current cruise speed is: %f m/s\n",
               setGlobalCruisespeed_.request.global_cruisespeed);
  } else {
    PRINT_ERROR("Set glogal cruise speed failed\n");
  }

  return setGlobalCruisespeed_.response.result;
}

DJI::OSDK::float32_t FFDS::MODULES::WpV2Operator::getGlobalCruiseSpeed(
    dji_osdk_ros::GetGlobalCruisespeed &getGlobalCruisespeed_) {
  waypointV2_get_global_cruisespeed_client =
      nh.serviceClient<dji_osdk_ros::GetGlobalCruisespeed>(
          "dji_osdk_ros/waypointV2_getGlobalCruisespeed");

  waypointV2_get_global_cruisespeed_client.call(getGlobalCruisespeed_);

  PRINT_INFO("Current cruise speed is: %f m/s\n",
             getGlobalCruisespeed_.response.global_cruisespeed);

  return getGlobalCruisespeed_.response.global_cruisespeed;
}
