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

void FFDS::APP::SingleFirePointTaskManager::attitudeSubCallback(
    const geometry_msgs::QuaternionStampedConstPtr &attitudeData) {
  attitude_data_ = *attitudeData;
}

void FFDS::APP::SingleFirePointTaskManager::gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition) {
  gps_position_ = *gpsPosition;
}

void FFDS::APP::SingleFirePointTaskManager::waypointV2MissionEventSubCallback(
    const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
        &waypointV2MissionEventPush) {
  waypoint_V2_mission_event_push_ = *waypointV2MissionEventPush;
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
void FFDS::APP::SingleFirePointTaskManager::waypointV2MissionStateSubCallback(
    const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
        &waypointV2MissionStatePush) {
  waypoint_V2_mission_state_push_ = *waypointV2MissionStatePush;
}

sensor_msgs::NavSatFix
FFDS::APP::SingleFirePointTaskManager::getHomeGPosAverage(int times) {
  sensor_msgs::NavSatFix homeGPos;

  for (int i = 0; (i < times) && ros::ok(); i++) {
    ros::spinOnce();
    homeGPos.latitude += gps_position_.latitude;
    homeGPos.longitude += gps_position_.longitude;
    homeGPos.altitude += gps_position_.altitude;

    if (TOOLS::isEquald(0.0, homeGPos.latitude) ||
        TOOLS::isEquald(0.0, homeGPos.longitude) ||
        TOOLS::isEquald(0.0, homeGPos.altitude)) {
      PRINT_WARN("zero in homeGPos");
    }
  }
  homeGPos.latitude = homeGPos.latitude / times;
  homeGPos.longitude = homeGPos.longitude / times;
  homeGPos.altitude = homeGPos.altitude / times;

  return homeGPos;
}

matrix::Eulerf FFDS::APP::SingleFirePointTaskManager::getInitAttAverage(
    int times) {
  /* NOTE: the quaternion from dji_osdk_ros to Eular angle is ENU! */
  /* NOTE: but the FlightTaskControl smaple node is NEU. Why they do this! :( */

  geometry_msgs::QuaternionStamped quat;

  for (int i = 0; (i < times) && ros::ok(); i++) {
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

void FFDS::APP::SingleFirePointTaskManager::initMission(
    dji_osdk_ros::InitWaypointV2Setting *initWaypointV2Setting_) {
  sensor_msgs::NavSatFix homeGPos = getHomeGPosAverage(100);
  PRINT_DEBUG("--------------------- Home Gpos ---------------------")
  PRINT_DEBUG("latitude: %f deg", homeGPos.latitude);
  PRINT_DEBUG("longitude: %f deg", homeGPos.longitude);
  PRINT_DEBUG("altitude: %f deg", homeGPos.altitude);

  matrix::Eulerf initAtt = getInitAttAverage(100);
  PRINT_DEBUG("--------------------- Init ENU Attitude ---------------------")
  PRINT_DEBUG("roll angle phi in ENU frame is: %f",
              TOOLS::Rad2Deg(initAtt.phi()));
  PRINT_DEBUG("pitch angle theta in ENU frame is: %f",
              TOOLS::Rad2Deg(initAtt.theta()));
  PRINT_DEBUG("yaw angle psi in ENU frame is: %f",
              TOOLS::Rad2Deg(initAtt.psi()));

  /* read the zigzag path shape parameters from yaml */
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/ZigzagPathShape.yaml";
  PRINT_INFO("Load zigzag shape from:%s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);

  int num = TOOLS::getParam(node, "num", 10);
  float len = TOOLS::getParam(node, "len", 40.0);
  float wid = TOOLS::getParam(node, "wid", 10.0);
  float height = TOOLS::getParam(node, "height", 15.0);

  MODULES::ZigzagPathPlanner pathPlanner(homeGPos, num, len, wid, height);

  initWaypointV2Setting_->request.polygonNum = 0;

  initWaypointV2Setting_->request.radius = 3;

  initWaypointV2Setting_->request.actionNum = 0;

  initWaypointV2Setting_->request.waypointV2InitSettings.repeatTimes = 1;

  initWaypointV2Setting_->request.waypointV2InitSettings.finishedAction =
      initWaypointV2Setting_->request.waypointV2InitSettings
          .DJIWaypointV2MissionFinishedGoHome;

  initWaypointV2Setting_->request.waypointV2InitSettings.maxFlightSpeed = 10;

  initWaypointV2Setting_->request.waypointV2InitSettings.autoFlightSpeed = 2;

  initWaypointV2Setting_->request.waypointV2InitSettings
      .exitMissionOnRCSignalLost = 1;

  initWaypointV2Setting_->request.waypointV2InitSettings.gotoFirstWaypointMode =
      initWaypointV2Setting_->request.waypointV2InitSettings
          .DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;

  initWaypointV2Setting_->request.waypointV2InitSettings.mission =
      pathPlanner.getWpV2Vec(true, true, initAtt.psi());

  initWaypointV2Setting_->request.waypointV2InitSettings.missTotalLen =
      initWaypointV2Setting_->request.waypointV2InitSettings.mission.size();
}

void FFDS::APP::SingleFirePointTaskManager::run() {
  MODULES::WpV2Operator wpV2Operator(nh);

  /* Step: 1 init the mission */
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initMission(&initWaypointV2Setting_);
  if (!wpV2Operator.initWaypointV2Setting(&initWaypointV2Setting_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* Step: 2 upload mission */
  dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
  if (!wpV2Operator.uploadWaypointV2Mission(&uploadWaypointV2Mission_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* Step: 3 start mission */
  dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
  if (!wpV2Operator.startWaypointV2Mission(&startWaypointV2Mission_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(30.0).sleep();

  /* Step: 4 call for the potential fire detecting */
  bool isPotentialFire = true;

  /* Step: 5 main loop */
  while (ros::ok() && (waypoint_V2_mission_state_push_.state != 0x6)) {
    if (!isPotentialFire) {
      continue;
    } else {
      dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
      if (!(wpV2Operator.pauseWaypointV2Mission(&pauseWaypointV2Mission_))) {
        PRINT_ERROR("Quit!");
        return;
      } else {
        PRINT_INFO("need to call the camera_gimbal! Later...")
        return;
      }
    }
  }

  return;
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "single_fire_point_task_manager_node");

  FFDS::APP::SingleFirePointTaskManager taskManager;
  taskManager.run();
  return 0;
}
