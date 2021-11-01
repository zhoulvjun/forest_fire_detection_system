/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraOperator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-31
 *
 *   @Description:
 *
 ******************************************************************************/

#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>

using namespace FFDS::MODULES;

void GimbalCameraOperator::singleFirePosIRCallback(
    const forest_fire_detection_system::SingleFirePosIR::ConstPtr
        &firePosition) {
  firePos = *firePosition;
};

bool GimbalCameraOperator::rotateGimbal(float setPosX, float setPosY,
                                        float timeOut, float tolErr) {
  PRINT_INFO("Start controlling the gimbal!");

  bool isTimeOut = false;
  bool isCtrlDone = false;

  ros::Time beginTime = ros::Time::now();
  float timeinterval;

  while (ros::ok() && (!isTimeOut) && (!isCtrlDone)) {

    ros::spinOnce();

    /* define error */
    float errX = setPosX - firePos.x;
    float errY = setPosY - firePos.y;
    if (errX <= tolErr && errY <= tolErr) {
      PRINT_INFO(
          "controling finish after %f seconds with x-error: %f, y-error: %f!",
          timeinterval, errX, errY);
      return true;
    }

    pidYaw.ctrl(errX);
    pidPitch.ctrl(errY);

    dji_osdk_ros::GimbalAction gimbalAction;
    gimbalAction.request.is_reset = false;
    gimbalAction.request.payload_index =
        static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
    gimbalAction.request.rotationMode = 0;
    gimbalAction.request.pitch = pidPitch.fullOutput();
    gimbalAction.request.roll = 0.0f;
    gimbalAction.request.yaw = pidYaw.fullOutput();
    gimbalAction.request.time = 1.0;
    gimbal_control_client.call(gimbalAction);

    timeinterval = TOOLS::getRosTimeInterval(beginTime);
    if (timeinterval >= timeOut) {
      isTimeOut = true;
      PRINT_WARN("control gimbal time out after %f seconds", timeinterval);
      return false;
    }
  }

  /* shutdown by keyboard */
  PRINT_WARN("stop gimbal control by keyboard...");
  return false;
}

void GimbalCameraOperator::zoomCamera() {}
