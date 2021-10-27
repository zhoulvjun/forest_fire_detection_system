/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_zigzag_planner.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-26
 *
 *   @Description:
 *
 ******************************************************************************/

#include <single_fire_point_task/modules/ZigzagPathPlanner.hpp>
#include <tools/GoogleEarthPath.hpp>
int main(int argc, char **argv) {
  sensor_msgs::NavSatFix home;
  home.latitude = 45.459074;
  home.longitude = -73.919047;
  home.altitude = 27.0;
  FFDS::ZigzagPathPlanner zigzagPlanner(home, 10, 100.0, 40, 15);
  std::vector<dji_osdk_ros::WaypointV2> waypointVec;

  float heading = FFDS::TOOLS::Deg2Rad(0);

  waypointVec = zigzagPlanner.getGPos(true, heading);

  GoogleEarthPath path("/home/ls/path1.kml", "path1");
  double longitude, latitude;

  for (int i = 0; i<waypointVec.size(); ++i){
    latitude = waypointVec[i].latitude;
    longitude = waypointVec[i].longitude;
    path.addPoint(longitude, latitude);
  } 

  return 0;
}
