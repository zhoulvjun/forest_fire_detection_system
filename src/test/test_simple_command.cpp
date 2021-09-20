/*******************************************************************************
*
*   Copyright (C) 2021 Lee Ltd. All rights reserved.
*
*   @Filename: test_simple_command.cpp
*
*   @Author: lee-shun
*
*   @Email: 2015097272@qq.com
*
*   @Date: 2021-09-19
*
*   @Description: 
*
******************************************************************************/

#include <test/test_simple_command.hpp>
int main(int argc, char** argv){
    ros::init(argc, argv, "test_simple_command_node");
    ros::NodeHandle nh;
    ROS_INFO("hello world!");
return 0;
}

