/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PathPlannerBase.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-26
 *
 *   @Description: The based class of the all kinds of path planner
 *
 ******************************************************************************/

#ifndef __PATHPLANNERBASE_HPP__
#define __PATHPLANNERBASE_HPP__

#include <iostream>
#include <sensor_msgs/NavSatFix.h>
#include <dji_osdk_ros/WaypointV2.h>

namespace FFDS {
namespace COMMOM {

class PathPlannerBase {
public:
  std::vector<dji_osdk_ros::WaypointV2> generateDJIWpV2Path();
  std::vector<sensor_msgs::NavSatFix> generateNavSaFixPath();
};

} // namespace COMMOM
} // namespace FFDS

#endif /* PATHPLANNERBASE_HPP */
