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
target_link_libraries(ZigzagPathPlanner
  ${catkin_LIBRARIES}
)

# Main node
add_executable(SingleFirePointTaskManager SingleFirePointTaskManager.cpp)
target_link_libraries(SingleFirePointTaskManager
  ${catkin_LIBRARIES}
)

