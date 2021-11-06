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

void GimbalCameraOperator::gimbalAttCallback(
    const geometry_msgs::Vector3Stamped::ConstPtr& att) {
  gimbalAtt = *att;
}

void GimbalCameraOperator::singleFirePosIRCallback(
    const forest_fire_detection_system::SingleFirePosIR::ConstPtr&
        firePosition) {
  firePosPix = *firePosition;
}

void GimbalCameraOperator::setGimbalActionDefault() {
  gimbalAction.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  gimbalAction.request.is_reset = false;
  gimbalAction.request.pitch = 0.0;
  gimbalAction.request.roll = 0.0f;
  gimbalAction.request.yaw = 0.0;
  gimbalAction.request.rotationMode = 0;
  gimbalAction.request.time = 1.0;
}

matrix::Vector3f GimbalCameraOperator::camera2NED(
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
/* TODO: need to sleep to wait until control done? */
bool GimbalCameraOperator::ctrlRotateGimbal(const float setPosXPix,
                                            const float setPosYPix,
                                            const int times,
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

    } else {
      if (ctrl_times > times) {
        PRINT_WARN("control gimbal times out after %d controlling!",
                   ctrl_times);
        return false;
      }

      float errX = setPosXPix - firePosPix.x;
      float errY = setPosYPix - firePosPix.y;
      PRINT_DEBUG("firePosition x %f", firePosPix.x);
      PRINT_DEBUG("firePosition y %f", firePosPix.y);

      if (fabs(errX) <= fabs(tolErrPix) && fabs(errY) <= fabs(tolErrPix)) {
        PRINT_INFO(
            "controling gimbal finish after %d times trying with x-error: "
            "%f pixel, y-error: %f pixel!",
            ctrl_times, errX, errY);
        return true;
      }

      PRINT_DEBUG("err Pitch:%f pixel", errX);
      PRINT_DEBUG("err Yaw:%f pixel", errY);

      pidYaw.ctrl(errX);
      pidPitch.ctrl(errY);

      /*NOTE: treat these attCam as degree */
      float d_pitchCam = pidPitch.getOutput();
      float d_yawCam = pidYaw.getOutput();

      PRINT_DEBUG("Pitch increment in Cam frame:%f deg ", d_pitchCam);
      PRINT_DEBUG("Yaw increment in Cam frame:%f deg", d_yawCam);

      /* NOTE: the gimbal x is pitch, y is roll, z is yaw, it's left hand
       * NOTE: rule??? YOU GOT BE KIDDING ME! */
      matrix::Vector3f d_attCam(d_pitchCam, 0.0f, d_yawCam);

      /* matrix::Vector3f d_attNED = camera2NED(d_attNED); */
      /* ROS_INFO_STREAM("d_attNED:" << d_attNED); */

      matrix::Vector3f AttStCam =
          d_attCam + matrix::Vector3f(gimbalAtt.vector.x, gimbalAtt.vector.y,
                                      gimbalAtt.vector.z);
      ROS_INFO_STREAM("AttStCam[0] pitch deg" << AttStCam(0));
      ROS_INFO_STREAM("AttStCam[1] roll deg" << AttStCam(1));
      ROS_INFO_STREAM("AttStCam[2] yaw deg" << AttStCam(2));

      setGimbalActionDefault();
      gimbalAction.request.is_reset = false;
      gimbalAction.request.pitch = AttStCam(0);
      gimbalAction.request.roll = AttStCam(1);
      gimbalAction.request.yaw = AttStCam(2);

      /* chang mode have a try */
      gimbalAction.request.rotationMode = 1;
      gimbalAction.request.time = 0.5;
      gimbalCtrlClient.call(gimbalAction);

      ctrl_times += 1;

      /* ros::Duration(3.0).sleep(); */
      int pause = std::cin.get();
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
bool GimbalCameraOperator::calRotateGimbal(
    const float setPosXPix, const float setPosYPix,
    const COMMON::IRCameraParams& H20TIr) {
  PRINT_INFO("Start controlling the gimbal using calculation!");

  ros::AsyncSpinner spinner(1);
  spinner.start();

  while (ros::ok()) {
    ros::spinOnce();

    if (!firePosPix.is_pot_fire) {
      PRINT_WARN("not stable potential fire, control restart!");
      continue;
    } else {
      double errX = setPosXPix - firePosPix.x;
      double errY = setPosYPix - firePosPix.y;

      float yawSetpoint =
          std::atan(-errX * H20TIr.eachPixInMM / H20TIr.equivalentFocalLength);
      float pitchSetpoint =
          std::atan(-errY * H20TIr.eachPixInMM / H20TIr.equivalentFocalLength);

      setGimbalActionDefault();
      gimbalAction.request.is_reset = false;
      gimbalAction.request.pitch = TOOLS::Rad2Deg(pitchSetpoint);
      gimbalAction.request.yaw = TOOLS::Rad2Deg(yawSetpoint) - 43.7; /* ?? */

      PRINT_DEBUG("yaw setpoint:%f, pitch setpoint:%f",
                  TOOLS::Rad2Deg(yawSetpoint), TOOLS::Rad2Deg(pitchSetpoint));

      /* from current point */
      gimbalAction.request.rotationMode = 1;
      gimbalAction.request.roll = 0.0f;
      gimbalAction.request.time = 1.0;

      gimbalCtrlClient.call(gimbalAction);

      break;
    }
  }

  return gimbalAction.response.result;
}

bool GimbalCameraOperator::resetGimbal() {
  setGimbalActionDefault();

  gimbalAction.request.is_reset = true;
  gimbalCtrlClient.call(gimbalAction);
  return gimbalAction.response.result;
}

bool GimbalCameraOperator::zoomCamera() { return false; }
