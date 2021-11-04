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

void GimbalCameraOperator::setGimbalActionDefault() {
  gimbalAction.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  gimbalAction.request.is_reset = false;
  gimbalAction.request.pitch = 0.0;
  gimbalAction.request.yaw = 0.0;
  gimbalAction.request.rotationMode = 0;
  gimbalAction.request.roll = 0.0f;
  gimbalAction.request.time = 1.0;
}

/**
 * @param[in]  x and y set position on the IR image, the controlling time and
 * finally control stop error.
 * @param[out]
 * @return
 * @ref
 * @see
 * @note control the gimbal rotate by the a PID controller, no need to use the
 * focal length, control several time according to the "timeOut"
 */
bool GimbalCameraOperator::rotateGimbalPID(float setPosX, float setPosY,
                                           float timeOutInS, float tolErr) {
  PRINT_INFO("Start controlling the gimbal!");

  ros::Time beginTime = ros::Time::now();
  float timeinterval;

  while (ros::ok()) {
    ros::spinOnce();

    if (!firePos.is_pot_fire) {

      PRINT_WARN("not stable potential fire, control restart!")
      pidYaw.reset();
      pidPitch.reset();
      beginTime = ros::Time::now();

    } else {

      float errX = setPosX - firePos.x;
      float errY = setPosY - firePos.y;

      if (fabs(errX) <= fabs(tolErr) && fabs(errY) <= fabs(tolErr)) {
        PRINT_INFO(
            "controling gimbal finish after %f seconds with x-error: "
            "%f, y-error: %f!",
            timeinterval, errX, errY);
        return true;
      }

      PRINT_DEBUG("err Pitch:%f ", errX);
      PRINT_DEBUG("err Yaw:%f ", errY);

      pidYaw.ctrl(-errX);
      pidPitch.ctrl(-errY);

      gimbalAction.request.is_reset = false;
      gimbalAction.request.pitch = pidPitch.fullOutput();
      gimbalAction.request.yaw = pidYaw.fullOutput();
      gimbalAction.request.rotationMode = 0;
      gimbalAction.request.roll = 0.0f;
      gimbalAction.request.time = 1.0;

      gimbal_control_client.call(gimbalAction);
    }

    timeinterval = TOOLS::getRosTimeInterval(beginTime);
    if (timeinterval >= timeOutInS) {
      PRINT_WARN("control gimbal time out after %f seconds", timeinterval);
      return false;
    }
  }

  /* shutdown by keyboard */
  PRINT_WARN("stop gimbal control by keyboard...");
  return false;
}

/**
 * @param[in]  x: position desired on the IR image along width, y : along height
 * @param[out]  void
 * @return void
 * @ref
 * @see
 * @note rotate the camera using the focal length and the image pixel length.
 * Rotate only one time.
 */
bool GimbalCameraOperator::rotateGimbalAngle(float setPosX, float setPosY) {
  return false;
}

bool GimbalCameraOperator::resetGimbal() {
  setGimbalActionDefault();
  gimbalAction.request.is_reset = true;
  gimbal_control_client.call(gimbalAction);
  return gimbalAction.response.result;
}

bool GimbalCameraOperator::zoomCamera() { return false; }
