/*******************************************************************************
*
*   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
*
*   @Filename: test_WpV2Operator.cpp
*
*   @Author: Shun Li
*
*   @Email: 2015097272@qq.com
*
*   @Date: 2021-10-25
*
*   @Description: rewirte the dji_osdk_ros/sample/waypointV2_node.cpp
*
******************************************************************************/

#include <test/test_WpV2Operator.hpp>

void gpsPositionSubCallback(const sensor_msgs::NavSatFix::ConstPtr& gpsPosition)
{
  gps_position_ = *gpsPosition;

  dbg(gps_position_.latitude);
  dbg(gps_position_.longitude);
  dbg(gps_position_.altitude);
}

void waypointV2MissionEventSubCallback(const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr& waypointV2MissionEventPush)
{
  waypoint_V2_mission_event_push_ = *waypointV2MissionEventPush;

  ROS_INFO("waypoint_V2_mission_event_push_.event ID :0x%x\n", waypoint_V2_mission_event_push_.event);

  if(waypoint_V2_mission_event_push_.event == 0x01)
  {
    ROS_INFO("interruptReason:0x%x\n", waypoint_V2_mission_event_push_.interruptReason);
  }
  if(waypoint_V2_mission_event_push_.event == 0x02)
  {
    ROS_INFO("recoverProcess:0x%x\n", waypoint_V2_mission_event_push_.recoverProcess);
  }
  if(waypoint_V2_mission_event_push_.event== 0x03)
  {
    ROS_INFO("finishReason:0x%x\n", waypoint_V2_mission_event_push_.finishReason);
  }

  if(waypoint_V2_mission_event_push_.event == 0x10)
  {
    ROS_INFO("current waypointIndex:%d\n", waypoint_V2_mission_event_push_.waypointIndex);
  }

  if(waypoint_V2_mission_event_push_.event == 0x11)
  {
    ROS_INFO("currentMissionExecNum:%d\n", waypoint_V2_mission_event_push_.currentMissionExecNum);
  }
}

void waypointV2MissionStateSubCallback(const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr& waypointV2MissionStatePush)
{
  waypoint_V2_mission_state_push_ = *waypointV2MissionStatePush;

  ROS_INFO("waypointV2MissionStateSubCallback");
  ROS_INFO("missionStatePushAck->commonDataVersion:%d\n",waypoint_V2_mission_state_push_.commonDataVersion);
  ROS_INFO("missionStatePushAck->commonDataLen:%d\n",waypoint_V2_mission_state_push_.commonDataLen);
  ROS_INFO("missionStatePushAck->data.state:0x%x\n",waypoint_V2_mission_state_push_.state);
  ROS_INFO("missionStatePushAck->data.curWaypointIndex:%d\n",waypoint_V2_mission_state_push_.curWaypointIndex);
  ROS_INFO("missionStatePushAck->data.velocity:%d\n",waypoint_V2_mission_state_push_.velocity);
}


bool runWaypointV2Mission(ros::NodeHandle &nh)
{
  int timeout = 1;
  bool result = false;

  waypointV2_mission_state_push_client = nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2Event>("dji_osdk_ros/waypointV2_subscribeMissionState");
  waypointV2_mission_event_push_client = nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2State>("dji_osdk_ros/waypointV2_subscribeMissionEvent");

  waypointV2EventSub = nh.subscribe("dji_osdk_ros/waypointV2_mission_event", 10, &waypointV2MissionEventSubCallback);
  waypointV2StateSub = nh.subscribe("dji_osdk_ros/waypointV2_mission_state", 10, &waypointV2MissionStateSubCallback);

  subscribeWaypointV2Event_.request.enable_sub = true;
  subscribeWaypointV2State_.request.enable_sub = true;

  get_drone_type_client.call(drone_type);
  if (drone_type.response.drone_type != static_cast<uint8_t>(dji_osdk_ros::Dronetype::M300))
  {
      ROS_DEBUG("This sample only supports M300!\n");
      return false;
  }

  // start publish the mission information
  waypointV2_mission_state_push_client.call(subscribeWaypointV2State_);
  waypointV2_mission_event_push_client.call(subscribeWaypointV2Event_);

    /*! init mission */

  result = initWaypointV2Setting(nh);
  if(!result)
  {
    return false;
  }
  sleep(timeout);

  /*! upload mission */
  result = uploadWaypointV2Mission(nh);
  if(!result)
  {
    return false;
  }
  sleep(timeout);

 /*! download mission */
  std::vector<dji_osdk_ros::WaypointV2> mission;
  result = downloadWaypointV2Mission(nh, mission);
  if(!result)
  {
    return false;
  }
  sleep(timeout);

  /*! upload  actions */
  result = uploadWaypointV2Action(nh);
  if(!result)
  {
    return false;
  }
  sleep(timeout);

  /*! start mission */
  result = startWaypointV2Mission(nh);
  if(!result)
  {
    return false;
  }
  sleep(20);

  /*! set global cruise speed */
  result = setGlobalCruiseSpeed(nh, 1.5);
  if(!result)
  {
    return false;
  }
  sleep(timeout);

  /*! get global cruise speed */
  DJI::OSDK::float32_t globalCruiseSpeed = 0;
  globalCruiseSpeed = getGlobalCruiseSpeed(nh);
  sleep(timeout);

  /*! pause the mission*/
  result = pauseWaypointV2Mission(nh);
  if(!result)
  {
    return false;
  }
  sleep(5);

  /*! resume the mission*/
  result = resumeWaypointV2Mission(nh);
  if(!result)
  {
    return false;
  }
  sleep(20);

return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_WpV2Operator_node");
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub = nh.subscribe("dji_osdk_ros/gps_position", 10, &gpsPositionSubCallback);
  auto obtain_ctrl_authority_client = nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
    "obtain_release_control_authority");

  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  obtainCtrlAuthority.request.enable_obtain = true;
  obtain_ctrl_authority_client.call(obtainCtrlAuthority);

  ros::Duration(1).sleep();
  ros::AsyncSpinner spinner(1);
  spinner.start();
  /* runWaypointV2Mission(nh); */

  ros::waitForShutdown();
}

