/*******************************************************************************
*
*   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
*
*   @Filename: single_fire_point_task_manager.cpp
*
*   @Author: Shun Li
*
*   @Email: 2015097272@qq.com
*
*   @Date: 2021-10-22
*
*   @Description: 
*
******************************************************************************/

#include <app/single_fire_point_task/SingleFirePointTaskManager.hpp>
using namespace FFDS::APP;

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "single_fire_point_task_manager_node");
  ros::NodeHandle nh;

  SingleFirePointTaskManager taskManager(nh);
  /* taskManager.run(); */
    return 0;
  }
