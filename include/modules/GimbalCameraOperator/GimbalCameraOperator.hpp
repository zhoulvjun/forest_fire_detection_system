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

#ifndef INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_
#define INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_

#include <dji_osdk_ros/GimbalAction.h>
#include <dji_osdk_ros/common_type.h>
#include <forest_fire_detection_system/SingleFirePosIR.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <PX4-Matrix/matrix/Matrix.hpp>
#include <common/CommonTypes.hpp>
#include <modules/BasicController/PIDController.hpp>
#include <tools/MathLib.hpp>
#include <tools/SystemLib.hpp>

namespace FFDS {

namespace MODULES {

class GimbalCameraOperator {
 public:
  GimbalCameraOperator() {
    singleFirePosIRSub =
        nh.subscribe("forest_fire_detection_system/single_fire_pos_ir_img", 10,
                     &GimbalCameraOperator::singleFirePosIRCallback, this);

    gimbalAttSub = nh.subscribe("dji_osdk_ros/gimbal_angle", 10,
                                &GimbalCameraOperator::gimbalAttCallback, this);

    gimbalCtrlClient =
        nh.serviceClient<dji_osdk_ros::GimbalAction>("gimbal_task_control");

    ros::Duration(3.0).sleep();
    PRINT_INFO("initialize GimbalCameraOperator done!");
  }

  bool ctrlRotateGimbal(const float setPosXPix, const float setPosYPix,
                        const int times, const float tolErrPix);
  bool resetGimbal();

  bool zoomCamera();
  bool resetCamera();

 private:
  ros::NodeHandle nh;
  ros::Subscriber singleFirePosIRSub;
  ros::Subscriber gimbalAttSub;
  ros::ServiceClient gimbalCtrlClient;

  dji_osdk_ros::GimbalAction gimbalAction;

  geometry_msgs::Vector3Stamped gimbalAtt;
  forest_fire_detection_system::SingleFirePosIR firePosPix;

  void singleFirePosIRCallback(
      const forest_fire_detection_system::SingleFirePosIR::ConstPtr&
          firePosition);

  void gimbalAttCallback(const geometry_msgs::Vector3Stamped::ConstPtr& att);

  void setGimbalActionDefault();

  matrix::Vector3f camera2NED(const matrix::Vector3f& d_attInCamera);

  PIDController pidYaw{0.01, 0.0, 0.0, false, false};
  PIDController pidPitch{0.01, 0.0, 0.0, false, false};
};

}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_
