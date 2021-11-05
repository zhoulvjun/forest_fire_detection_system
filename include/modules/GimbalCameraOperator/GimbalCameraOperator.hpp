/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraOperator.hpp
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

#ifndef __GIMBALCAMERAOPERATOR_HPP__
#define __GIMBALCAMERAOPERATOR_HPP__

#include <dji_osdk_ros/GimbalAction.h>
#include <dji_osdk_ros/common_type.h>
#include <forest_fire_detection_system/SingleFirePosIR.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <common/CommonTypes.hpp>
#include <modules/BasicController/IncPIDController.hpp>
#include <modules/BasicController/PIDController.hpp>
#include <tools/MathLib.hpp>
#include <tools/SystemLib.hpp>
#include <PX4-Matrix/matrix/Matrix.hpp>

namespace FFDS {

namespace MODULES {

class GimbalCameraOperator {
 public:
  GimbalCameraOperator() {
    singleFirePosIRSub =
        nh.subscribe("forest_fire_detection_system/single_fire_pos_ir_img", 10,
                     &GimbalCameraOperator::singleFirePosIRCallback, this);

    gimbalCtrlClient =
        nh.serviceClient<dji_osdk_ros::GimbalAction>("gimbal_task_control");

    ros::Duration(3.0).sleep();
    PRINT_INFO("initialize GimbalCameraOperator done!");
  };

  bool ctrlRotateGimbal(const float setPosXPix, const float setPosYPix,
                        const float timeOutInS, const float tolErrPix);
  bool calRotateGimbal(const float setPosXPix, const float setPosYPix,
                       const COMMON::IRCameraParams& H20TIr);
  bool resetGimbal();

  bool zoomCamera();
  bool resetCamera();

 private:
  ros::NodeHandle nh;
  ros::Subscriber singleFirePosIRSub;
  ros::ServiceClient gimbalCtrlClient;

  dji_osdk_ros::GimbalAction gimbalAction;
  forest_fire_detection_system::SingleFirePosIR firePosPix;

  void singleFirePosIRCallback(
      const forest_fire_detection_system::SingleFirePosIR::ConstPtr&
          firePosition);

  void setGimbalActionDefault();

  PIDController pidYaw{0.01, 0.0, 0.0, false, false};
  PIDController pidPitch{0.01, 0.0, 0.0, false, false};
};

}  // namespace MODULES
}  // namespace FFDS

#endif /* GIMBALCAMERAOPERATOR_HPP */
