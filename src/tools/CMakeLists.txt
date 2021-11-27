#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: CMakeLists.txt
#
#   @Author: Shun Li
#
#   @Date: 2021-11-23
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------


add_executable(AlianRGB_IR AlianRGB_IR.cpp)
target_link_libraries(
  AlianRGB_IR
  ${OpenCV_LIBRARIES}
)
target_link_libraries(
  AlianRGB_IR
  ${cv_bridge_LIBRARIES}
)
target_link_libraries(
  AlianRGB_IR
  ${catkin_LIBRARIES}
)

add_executable(ToggleVehicleVideo ToggleVehicleVideo.cpp)
target_link_libraries(
  ToggleVehicleVideo
  RGB_IRSeperator
)
target_link_libraries(
  ToggleVehicleVideo
  ${catkin_LIBRARIES}
)
