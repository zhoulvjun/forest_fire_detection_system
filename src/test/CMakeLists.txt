add_executable(test_simple_command test_simple_command.cpp)
target_link_libraries(test_simple_command
  ${catkin_LIBRARIES}
)

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
