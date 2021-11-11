#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: CMakeLists.txt
#
#   @Author: Shun Li
#
#   @Date: 2021-10-22
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------

add_executable(
  SingleFirePointTaskManager
  single_fire_point_task/SingleFirePointTaskManager.cpp
)
target_link_libraries(
  SingleFirePointTaskManager
  WpV2Operator
  ZigzagPathPlanner
  GimbalCameraOperator
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
