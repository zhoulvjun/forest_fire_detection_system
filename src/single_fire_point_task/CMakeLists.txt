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

# Path Planner
add_library(ZigzagPathPlanner modules/ZigzagPathPlanner.cpp)

# Main node
add_executable(single_fire_point_task_manager single_fire_point_task_manager.cpp)
target_link_libraries(single_fire_point_task_manager
  ${catkin_LIBRARIES}
)

