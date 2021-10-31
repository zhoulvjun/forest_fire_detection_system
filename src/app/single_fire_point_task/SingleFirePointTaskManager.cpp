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
#include <tools/PrintControl/PrintCtrlImp.h>

using namespace FFDS::APP;

void SingleFirePointTaskManager::attitudeSubCallback(
    const geometry_msgs::QuaternionStampedConstPtr &attitudeData)
{

  attitude_data_ = *attitudeData;
}

void SingleFirePointTaskManager::gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition)
{

  gps_position_ = *gpsPosition;
}

void SingleFirePointTaskManager::waypointV2MissionEventSubCallback(
    const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
        &waypointV2MissionEventPush)
{

  waypoint_V2_mission_event_push_ = *waypointV2MissionEventPush;

  ROS_INFO("waypoint_V2_mission_event_push_.event ID :0x%x\n",
           waypoint_V2_mission_event_push_.event);

  if (waypoint_V2_mission_event_push_.event == 0x01)
  {
    ROS_INFO("interruptReason:0x%x\n",
             waypoint_V2_mission_event_push_.interruptReason);
  }
  if (waypoint_V2_mission_event_push_.event == 0x02)
  {
    ROS_INFO("recoverProcess:0x%x\n",
             waypoint_V2_mission_event_push_.recoverProcess);
  }
  if (waypoint_V2_mission_event_push_.event == 0x03)
  {
    ROS_INFO("finishReason:0x%x\n",
             waypoint_V2_mission_event_push_.finishReason);
  }

  if (waypoint_V2_mission_event_push_.event == 0x10)
  {
    ROS_INFO("current waypointIndex:%d\n",
             waypoint_V2_mission_event_push_.waypointIndex);
  }

  if (waypoint_V2_mission_event_push_.event == 0x11)
  {
    ROS_INFO("currentMissionExecNum:%d\n",
             waypoint_V2_mission_event_push_.currentMissionExecNum);
  }
}

/*
 * 0x0:ground station not start.
 * 0x1:mission prepared.
 * 0x2:enter mission.
 * 0x3:execute flying route mission.
 * 0x4:pause state.
 * 0x5:enter mission after ending pause.
 * 0x6:exit mission.
 * */
void SingleFirePointTaskManager::waypointV2MissionStateSubCallback(
    const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
        &waypointV2MissionStatePush)
{

  waypoint_V2_mission_state_push_ = *waypointV2MissionStatePush;

  ROS_INFO("waypointV2MissionStateSubCallback");
  ROS_INFO("missionStatePushAck->commonDataVersion:%d\n",
           waypoint_V2_mission_state_push_.commonDataVersion);
  ROS_INFO("missionStatePushAck->commonDataLen:%d\n",
           waypoint_V2_mission_state_push_.commonDataLen);
  ROS_INFO("missionStatePushAck->data.state:0x%x\n",
           waypoint_V2_mission_state_push_.state);
  ROS_INFO("missionStatePushAck->data.curWaypointIndex:%d\n",
           waypoint_V2_mission_state_push_.curWaypointIndex);
  ROS_INFO("missionStatePushAck->data.velocity:%d\n",
           waypoint_V2_mission_state_push_.velocity);
}

sensor_msgs::NavSatFix
SingleFirePointTaskManager::getHomeGPosAverage(int times)
{

  sensor_msgs::NavSatFix homeGPos;

  for (int i = 0; (i < times) && ros::ok(); i++)
  {
    ros::spinOnce();
    homeGPos.latitude += gps_position_.latitude;
    homeGPos.longitude += gps_position_.longitude;
    homeGPos.altitude += gps_position_.altitude;

    if (TOOLS::isEquald(0.0, homeGPos.latitude) ||
        TOOLS::isEquald(0.0, homeGPos.longitude) ||
        TOOLS::isEquald(0.0, homeGPos.altitude))
    {
      PRINT_WARN("zero in homeGPos");
    }
  }
  homeGPos.latitude = homeGPos.latitude / times;
  homeGPos.longitude = homeGPos.longitude / times;
  homeGPos.altitude = homeGPos.altitude / times;

  return homeGPos;
}

matrix::Eulerf SingleFirePointTaskManager::getInitAttAverage(int times)
{

  /* NOTE: the quaternion from dji_osdk_ros to Eular angle is ENU! */
  /* NOTE: but the FlightTaskControl smaple node is NEU. Why they do this! :( */

  geometry_msgs::QuaternionStamped quat;

  for (int i = 0; (i < times) && ros::ok(); i++)
  {
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

  matrix::Quaternionf average_quat(quat.quaternion.w, quat.quaternion.x,
                                   quat.quaternion.y, quat.quaternion.z);

  return matrix::Eulerf(average_quat);
}

void SingleFirePointTaskManager::initMission(
    dji_osdk_ros::InitWaypointV2Setting &initWaypointV2Setting_)
{

  sensor_msgs::NavSatFix homeGPos = getHomeGPosAverage(100);
  PRINT_INFO("--------------------- Home Gpos ---------------------")
  ROS_INFO_STREAM("latitude:" << homeGPos.latitude);
  ROS_INFO_STREAM("longitude:" << homeGPos.longitude);
  ROS_INFO_STREAM("altitude:" << homeGPos.altitude);

  matrix::Eulerf initAtt = getInitAttAverage(100);
  PRINT_INFO("--------------------- Init ENU Attitude ---------------------")
  ROS_INFO_STREAM(
      "roll angle phi in ENU frame is:" << TOOLS::Rad2Deg(initAtt.phi()));
  ROS_INFO_STREAM(
      "pitch angle theta in ENU frame is:" << TOOLS::Rad2Deg(initAtt.theta()));
  ROS_INFO_STREAM(
      "yaw angle psi in ENU frame is:" << TOOLS::Rad2Deg(initAtt.psi()));

  MODULES::ZigzagPathPlanner pathPlanner(homeGPos, 10, 100.0, 40, 15);

  initWaypointV2Setting_.request.polygonNum = 0;

  initWaypointV2Setting_.request.radius = 3;

  initWaypointV2Setting_.request.actionNum = 0;

  initWaypointV2Setting_.request.waypointV2InitSettings.repeatTimes = 1;

  initWaypointV2Setting_.request.waypointV2InitSettings.finishedAction =
      initWaypointV2Setting_.request.waypointV2InitSettings
          .DJIWaypointV2MissionFinishedGoHome;

  initWaypointV2Setting_.request.waypointV2InitSettings.maxFlightSpeed = 10;

  initWaypointV2Setting_.request.waypointV2InitSettings.autoFlightSpeed = 2;

  initWaypointV2Setting_.request.waypointV2InitSettings
      .exitMissionOnRCSignalLost = 1;

  initWaypointV2Setting_.request.waypointV2InitSettings.gotoFirstWaypointMode =
      initWaypointV2Setting_.request.waypointV2InitSettings
          .DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;

  initWaypointV2Setting_.request.waypointV2InitSettings.mission =
      pathPlanner.getWpV2Vec(true, true, initAtt.psi());

  initWaypointV2Setting_.request.waypointV2InitSettings.missTotalLen =
      initWaypointV2Setting_.request.waypointV2InitSettings.mission.size();
}

void SingleFirePointTaskManager::run()
{

  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  obtainCtrlAuthority.request.enable_obtain = true;
  obtain_ctrl_authority_client.call(obtainCtrlAuthority);

  MODULES::WpV2Operator wpV2Operator(nh);

  /* Step: 1 init the mission */
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initMission(initWaypointV2Setting_);
  if (!wpV2Operator.initWaypointV2Setting(initWaypointV2Setting_))
  {
    PRINT_ERROR("Quit!");
    return;
  }

  /* Step: 2 upload mission */
  dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
  if (!wpV2Operator.uploadWaypointV2Mission(uploadWaypointV2Mission_))
  {
    PRINT_ERROR("Quit!");
    return;
  }

  /* Step: 3 start mission */
  dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
  if (!wpV2Operator.startWaypointV2Mission(startWaypointV2Mission_))
  {
    PRINT_ERROR("Quit!");
    return;
  }

  /* Step: 4 call for the potential fire detecting */
  bool isPotentialFire = true;

  /* Step: 5 main loop */
  while (ros::ok() && (waypoint_V2_mission_state_push_.state != 0x6))
  {

    if (!isPotentialFire)
    {
      continue;
    }
    else
    {

      dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
      if (!wpV2Operator.pauseWaypointV2Mission(pauseWaypointV2Mission_))
      {
        PRINT_ERROR("Quit!");
        return;
      }

      PRINT_INFO("need to call the camera_gimbal!")
    }
  }

  return;
}

int main(int argc, char *argv[])
{

  ros::init(argc, argv, "single_fire_point_task_manager_node");

  SingleFirePointTaskManager taskManager;
  taskManager.run();
  return 0;
}
