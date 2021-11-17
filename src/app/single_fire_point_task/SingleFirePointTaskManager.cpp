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

FFDS::APP::SingleFirePointTaskManager::SingleFirePointTaskManager() {
  task_control_client =
      nh.serviceClient<dji_osdk_ros::FlightTaskControl>("/flight_task_control");
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
  singleFirePosIRSub =
      nh.subscribe("forest_fire_detection_system/single_fire_pos_ir_img", 10,
                   &SingleFirePointTaskManager::singleFirePosIRCallback, this);

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

  ros::Duration(3.0).sleep();
  PRINT_INFO("initializing Done");
}

FFDS::APP::SingleFirePointTaskManager::~SingleFirePointTaskManager() {
  obtainCtrlAuthority.request.enable_obtain = false;
  obtain_ctrl_authority_client.call(obtainCtrlAuthority);
  if (obtainCtrlAuthority.response.result) {
    PRINT_INFO("release control authority!");
  } else {
    PRINT_ERROR("can NOT release control authority!");
  }
}

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

void FFDS::APP::SingleFirePointTaskManager::singleFirePosIRCallback(
    const forest_fire_detection_system::SingleFirePosIR::ConstPtr &sfPos) {
  signleFirePos = *sfPos;
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
    dji_osdk_ros::InitWaypointV2Setting *initWaypointV2SettingPtr) {
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

  initWaypointV2SettingPtr->request.polygonNum = 0;

  initWaypointV2SettingPtr->request.radius = 3;

  initWaypointV2SettingPtr->request.actionNum = 0;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.repeatTimes = 1;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.finishedAction =
      initWaypointV2SettingPtr->request.waypointV2InitSettings
          .DJIWaypointV2MissionFinishedGoHome;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.maxFlightSpeed = 5;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.autoFlightSpeed = 1;

  initWaypointV2SettingPtr->request.waypointV2InitSettings
      .exitMissionOnRCSignalLost = 1;

  initWaypointV2SettingPtr->request.waypointV2InitSettings
      .gotoFirstWaypointMode =
      initWaypointV2SettingPtr->request.waypointV2InitSettings
          .DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.mission =
      pathPlanner.getWpV2Vec(true, initAtt.psi());

  initWaypointV2SettingPtr->request.waypointV2InitSettings.missTotalLen =
      initWaypointV2SettingPtr->request.waypointV2InitSettings.mission.size();
}

void FFDS::APP::SingleFirePointTaskManager::goHomeLand() {
  PRINT_INFO("going home now");
  control_task.request.task =
      dji_osdk_ros::FlightTaskControl::Request::TASK_GOHOME;
  task_control_client.call(control_task);
  if (control_task.response.result == true) {
    PRINT_INFO("go home successful");
  } else {
    PRINT_WARN("go home failed.");
  }

  control_task.request.task =
      dji_osdk_ros::FlightTaskControl::Request::TASK_LAND;
  PRINT_INFO(
      "Landing request sending ... need your confirmation on the remoter!");
  task_control_client.call(control_task);
  if (control_task.response.result == true) {
    PRINT_INFO("land task successful");
  } else {
    PRINT_ERROR("land task failed.");
  }
}

void FFDS::APP::SingleFirePointTaskManager::run() {
  FFDS::MODULES::WpV2Operator wpV2Operator;
  FFDS::MODULES::GimbalCameraOperator gcOperator;

  /* Step: 0 reset the camera and gimbal */
  if (gcOperator.resetCameraZoom() && gcOperator.resetGimbal()) {
    PRINT_INFO("reset camera and gimbal successfully!")
  } else {
    PRINT_WARN("reset camera and gimbal failed!")
  }

  /* Step: 1 init the mission, create the basic waypointV2 vector... */
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initMission(&initWaypointV2Setting_);
  if (!wpV2Operator.initWaypointV2Setting(&initWaypointV2Setting_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* Step: 2 upload mission, empty srv works */
  dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
  if (!wpV2Operator.uploadWaypointV2Mission(&uploadWaypointV2Mission_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  PRINT_INFO("WpV2Mission init finish, are you ready to start? y/n");
  char inputConfirm;
  std::cin >> inputConfirm;
  if (inputConfirm == 'n') {
    PRINT_WARN("exist!");
    return;
  }

  /* Step: 3 start mission, empty srv works */
  dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
  if (!wpV2Operator.startWaypointV2Mission(&startWaypointV2Mission_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  /**
   * Step: 4 main loop
   * 1. "break" in the following while-loop is only for serious error and task
   *exit...
   * 2. 0x6 == exit waypointV2 mission
   **/
  PRINT_INFO("start find the potential fire~");
  int isPotFireNum = 0;
  while (ros::ok() && (waypoint_V2_mission_state_push_.state != 0x6)) {
    ros::spinOnce();
    if (!signleFirePos.is_pot_fire) {
      isPotFireNum = 0;
      continue;
    } else {
      isPotFireNum += 1;
    }

    if (isPotFireNum < 10) {
      PRINT_INFO(
          "potential fire FOUND %d times! NOT stable enough, check again!",
          isPotFireNum);
      ros::Rate(2).sleep();
    } else {
      PRINT_INFO("potential fire FOUND %d times! call to pause the mission!",
                 isPotFireNum);
      dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
      if (!(wpV2Operator.pauseWaypointV2Mission(&pauseWaypointV2Mission_))) {
        PRINT_ERROR(
            "failed to pause the mission, please use remoter to cancle!");
        break;
      }

      PRINT_INFO(
          "pause mission successfully, call for gimbal and camera to focus...")

      if (gcOperator.ctrlRotateGimbal(10, 20.0)) {
        PRINT_INFO("rotate done! zoom the camera!")
      } else {
        PRINT_WARN("failed to rotate camera, use bigger tolerance!");
        if (gcOperator.ctrlRotateGimbal(10, 30.0)) {
          PRINT_INFO("rotate done wuth bigger tolerance!");
        } else {
          PRINT_WARN(
              "failed to rotate camera with bigger tolerance! ZOOM anyway!");
        }
      }

      if (gcOperator.setCameraZoom(5.0)) {
        PRINT_INFO("zoom done! call for further detecting!")
      } else {
        PRINT_WARN("failed to zoom camera, further detect anyway~");
      }

      /* call for detecting */
      ros::Duration(15.0).sleep();

      PRINT_INFO("further detecting done! reset gimbal and camera!")
      if (gcOperator.resetGimbal()) {
        PRINT_INFO("reset gimbal done! zoom the camera!")
      } else {
        PRINT_WARN("failed to reset gimbal!");
      }

      if (gcOperator.resetCameraZoom()) {
        PRINT_INFO("reset zoom done!")
      } else {
        PRINT_WARN("failed to zoom camera");
      }

      PRINT_INFO("go home and land...")
      dji_osdk_ros::StopWaypointV2Mission stopWaypointV2Mission_;
      if (!wpV2Operator.stopWaypointV2Mission(&stopWaypointV2Mission_)) {
        PRINT_ERROR("can not stop waypointV2 mission, please use the remoter!");
        break;
      }

      goHomeLand();
      break;
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
