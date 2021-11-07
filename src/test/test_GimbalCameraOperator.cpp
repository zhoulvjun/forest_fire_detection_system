/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_GimbalCameraOperator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-02
 *
 *   @Description:
 *
 ******************************************************************************/
#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_GimbalCameraOperator_node");
  FFDS::MODULES::GimbalCameraOperator gimbalCameraOperator;
  FFDS::COMMON::CameraParams H20t(
      "/home/ls/catkin_ws/src/forest_fire_detection_system/config/"
      "H20T_IR_Camera.yaml");

  PRINT_INFO(
      "control test result: %d",
      gimbalCameraOperator.ctrlRotateGimbal(
          H20t.splitImgWidthPix / 2, H20t.splitImgHeightPix / 2, 10, 30));

  return 0;
}
