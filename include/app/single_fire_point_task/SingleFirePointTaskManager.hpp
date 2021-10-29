/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SingleFirePointTaskManager.hpp
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

#ifndef __SINGLEFIREPOINTTASKMANAGER_HPP__
#define __SINGLEFIREPOINTTASKMANAGER_HPP__

#include <ros/ros.h>
#include <modules/PathPlanner/ZigzagPathPlanner.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>

namespace FFDS {
namespace APP {

class SingleFirePointTaskManager {
  private:
    ros::NodeHandle nh;
    MODULES::ZigzagPathPlanner pathPlanner;
    MODULES::WpV2Operator wpV2Operator;
  public:

    SingleFirePointTaskManager(ros::NodeHandle &h):nh(h), wpV2Operator(h){
      std::cout<<"initialized SingleFirePointTaskManager, nh address:"<< &nh <<std::endl;
    };

    int run();

};
} // namespace APP
} // namespace FFDS

#endif /* SINGLEFIREPOINTTASKMANAGER_HPP */
