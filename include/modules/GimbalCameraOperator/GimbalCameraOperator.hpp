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
#include <forest_fire_detection_system/SingleFirePosIR.h>
#include <boost/exception/exception.hpp>
#include <modules/BasicController/IncPIDController.hpp>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlImp.h>
#include <tools/SystemLib.hpp>
#include <dji_osdk_ros/common_type.h>
#include <tools/MathLib.hpp>

namespace FFDS {

namespace MODULES {

class GimbalCameraOperator {
public:
  GimbalCameraOperator()
      : pidYaw(0.01, 0.00, 0.00), pidPitch(0.01, 0.00, 0.00) {

    singleFirePosIRSub =
        nh.subscribe("forest_fire_detection_system/single_fire_pos_ir_img", 10,
                     &GimbalCameraOperator::singleFirePosIRCallback, this);

     gimbal_control_client = nh.serviceClient<dji_osdk_ros::GimbalAction>("gimbal_task_control");

    ros::Duration(3.0).sleep();
    PRINT_INFO("initialize GimbalCameraOperator done!");
  };

  bool rotateGimbalPID(float setPosX, float setPosY, float timeOut, float tolErr);
  bool rotateGimbalGeometry();
  bool resetGimbal();

  void zoomCamera();
  void resetCamera();

private:
  ros::NodeHandle nh;
  ros::Subscriber singleFirePosIRSub;
  ros::ServiceClient gimbal_control_client;

  forest_fire_detection_system::SingleFirePosIR firePos;

  void singleFirePosIRCallback(
      const forest_fire_detection_system::SingleFirePosIR::ConstPtr
          &firePosition);

  IncPIDController pidYaw;
  IncPIDController pidPitch;
};

} // namespace MODULES
} // namespace FFDS

#endif /* GIMBALCAMERAOPERATOR_HPP */
