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
  gimbalCameraOperator.rotateGimbalPID(300, 200, 30.0, 10.0);

  return 0;
}
