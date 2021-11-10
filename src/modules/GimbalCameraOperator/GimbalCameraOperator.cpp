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

void FFDS::MODULES::GimbalCameraOperator::gimbalAttCallback(
    const geometry_msgs::Vector3Stamped::ConstPtr& att) {
  gimbalAtt = *att;
}

void FFDS::MODULES::GimbalCameraOperator::singleFirePosIRCallback(
    const forest_fire_detection_system::SingleFirePosIR::ConstPtr&
        firePosition) {
  firePosPix = *firePosition;
}

void FFDS::MODULES::GimbalCameraOperator::setGimbalActionDefault() {
  gimbalAction.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  gimbalAction.request.is_reset = false;
  gimbalAction.request.pitch = 0.0;
  gimbalAction.request.roll = 0.0f;
  gimbalAction.request.yaw = 0.0;
  gimbalAction.request.rotationMode = 0;
  gimbalAction.request.time = 1.0;
}

matrix::Vector3f FFDS::MODULES::GimbalCameraOperator::camera2NED(
    const matrix::Vector3f& d_attInCamera) {
  float phi = TOOLS::Deg2Rad(gimbalAtt.vector.x);   /* roll angle */
  float theta = TOOLS::Deg2Rad(gimbalAtt.vector.y); /* pitch angle */
  float psi = TOOLS::Deg2Rad(gimbalAtt.vector.z);   /* yaw angle */

  float convert[3][3] = {{1, sin(phi) * tan(theta), cos(phi) * tan(theta)},
                         {0, cos(phi), -sin(phi)},
                         {0, sin(phi) / cos(theta), cos(phi) / cos(theta)}};

  matrix::Matrix3f eularMatrix(convert);

  return eularMatrix * d_attInCamera;
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
bool FFDS::MODULES::GimbalCameraOperator::ctrlRotateGimbal(
    const float setPosXPix, const float setPosYPix, const int times,
    const float tolErrPix) {
  PRINT_INFO("Start controlling the gimbal using controller!");

  int ctrl_times = 0;
  while (ros::ok()) {
    ros::spinOnce();

    if (!firePosPix.is_pot_fire) {
      pidYaw.reset();
      pidPitch.reset();
      ctrl_times = 0;
      PRINT_WARN("not stable potential fire, control restart!")
      ros::Duration(1.0).sleep();
      continue;
    } else {
      if (ctrl_times > times) {
        PRINT_WARN("control gimbal times out after %d controlling!",
                   ctrl_times);
        return false;
      }

      PRINT_INFO("current control times: %d, tolerance: %d", ctrl_times, times);

      float errX = setPosXPix - firePosPix.x;
      float errY = setPosYPix - firePosPix.y;
      PRINT_DEBUG("err Yaw:%f pixel", errX);
      PRINT_DEBUG("err Pitch:%f pixel", errY);

      if (fabs(errX) <= fabs(tolErrPix) && fabs(errY) <= fabs(tolErrPix)) {
        PRINT_INFO(
            "controling gimbal finish after %d times trying with x-error: "
            "%f pixel, y-error: %f pixel!",
            ctrl_times, errX, errY);
        return true;
      }

      /* +x error -> - inc yaw */
      /* +y error -> + inc pitch */
      pidYaw.ctrl(-errX);
      pidPitch.ctrl(errY);

      /*NOTE: treat these attCam as degree */
      float d_pitchCam = pidPitch.getOutput();
      float d_yawCam = pidYaw.getOutput();

      PRINT_DEBUG("Pitch increment in Cam frame:%f deg ", d_pitchCam);
      PRINT_DEBUG("Yaw increment in Cam frame:%f deg", d_yawCam);

      /* NOTE: the gimbal x is pitch, y is roll, z is yaw, it's left hand
       * NOTE: rule??? YOU GOT BE KIDDING ME! */
      matrix::Vector3f d_attCam(d_pitchCam, 0.0f, d_yawCam);

      setGimbalActionDefault();
      gimbalAction.request.is_reset = false;
      gimbalAction.request.pitch = d_attCam(0);
      gimbalAction.request.roll = d_attCam(1);
      gimbalAction.request.yaw = d_attCam(2);

      /* 0 for incremental mode, 1 for absolute mode */
      gimbalAction.request.rotationMode = 0;
      gimbalAction.request.time = 0.5;
      gimbalCtrlClient.call(gimbalAction);

      ctrl_times += 1;

      ros::Duration(1.0).sleep();
    }
  }

  /* shutdown by keyboard */
  PRINT_WARN("stop gimbal control by keyboard...");
  return false;
}

bool FFDS::MODULES::GimbalCameraOperator::resetGimbal() {
  setGimbalActionDefault();

  gimbalAction.request.is_reset = true;
  gimbalCtrlClient.call(gimbalAction);
  return gimbalAction.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::setCameraZoom(const float factor) {
  cameraSetZoomPara.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraSetZoomPara.request.factor = factor;
  cameraSetZoomParaClient.call(cameraSetZoomPara);
  return cameraSetZoomPara.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetCameraZoom() {
  return setCameraZoom(2.0);
}

bool FFDS::MODULES::GimbalCameraOperator::setCameraFocusePoint(const float x,
                                                               const float y) {
  cameraFocusPoint.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraFocusPoint.request.x = x;
  cameraFocusPoint.request.y = y;
  cameraSetFocusPointClient.call(cameraFocusPoint);
  return cameraFocusPoint.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetCameraFocusePoint() {
  return setCameraFocusePoint(0.5, 0.5);
}

bool FFDS::MODULES::GimbalCameraOperator::setTapZoomPoint(
    const float multiplier, const float x, const float y) {
  cameraTapZoomPoint.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraTapZoomPoint.request.multiplier = multiplier;
  cameraTapZoomPoint.request.x = x;
  cameraTapZoomPoint.request.y = y;
  cameraSetTapZoomPointClient.call(cameraTapZoomPoint);
  return cameraFocusPoint.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetTapZoomPoint() {
  return setTapZoomPoint(2.0, 0.5, 0.5);
}
