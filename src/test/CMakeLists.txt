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

# test the control command
add_executable(test_simple_command test_simple_command.cpp)
target_link_libraries(test_simple_command
  ${catkin_LIBRARIES}
)

# test the WpV2Operator
add_executable(test_WpV2Operator test_WpV2Operator.cpp)
target_link_libraries(test_WpV2Operator
  ${catkin_LIBRARIES}
)
target_link_libraries(test_WpV2Operator
  WpV2Operator
)

# test zigzagPlanner
add_executable(test_zigzag_planner test_zigzag_planner.cpp)
target_link_libraries(test_zigzag_planner
  ${catkin_LIBRARIES}
)
target_link_libraries(test_zigzag_planner
  ZigzagPathPlanner
)

# test GimbalCameraOperator
add_executable(test_GimbalCameraOperator test_GimbalCameraOperator.cpp)
target_link_libraries(test_GimbalCameraOperator
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
target_link_libraries(test_GimbalCameraOperator
  GimbalCameraOperator
)

# test the getParam yaml
add_executable(test_getParam test_getParam.cpp)
target_link_libraries(test_getParam
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
