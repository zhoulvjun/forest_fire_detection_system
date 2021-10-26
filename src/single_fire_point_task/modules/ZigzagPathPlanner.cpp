/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ZigzagPathPlanner.cpp
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

using namespace FFDS;

void ZigzagPathPlanner::calLocalPos() {
  float each_len = zigzagLen / zigzagNum;
  int point_num = 2 * (zigzagNum + 1);
  COMMON::LocalPosition pos;

  bool is_lower_left = true;
  bool is_upper_left = false;
  bool is_lower_right = false;
  bool is_upper_right = false;

  for (int i = 0; i < point_num - 1; ++i) {

    pos.z = zigzagHeight;

    if (is_lower_left) {
      pos.x += 0.0;
      pos.y += zigzagWid;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = true;
      is_upper_right = false;
    } else if (is_lower_right) {
      pos.x += each_len;
      pos.y += 0.0;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = true;
    } else if (is_upper_right) {
      pos.x += 0.0;
      pos.y += -zigzagWid;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = true;
      is_lower_right = false;
      is_upper_right = false;
    } else if (is_upper_left) {
      pos.x += each_len;
      pos.y += 0.0;
      LocalPosVec.push_back(pos);

      is_lower_left = true;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = false;
    } else {
      ROS_ERROR_STREAM("the bool is wrong!");
    }
  }
}

void ZigzagPathPlanner::HEarth2Earth(float heading) {

  for (int i = 0; i < LocalPosVec.size(); ++i) {

    LocalPosVec[i].x =
        LocalPosVec[i].x * cos(heading) - LocalPosVec[i].y * sin(heading);

    LocalPosVec[i].y =
        LocalPosVec[i].x * sin(heading) + LocalPosVec[i].y * cos(heading);

    LocalPosVec[i].z = LocalPosVec[i].z;
  }
}

std::vector<dji_osdk_ros::WaypointV2> &
ZigzagPathPlanner::getGPos(bool useInitHeadDirection, float headingRad) {

  /* Step: 1 generate the local zigzag pos */
  calLocalPos();

  /* Step: 2 if HeadEarth to Earth? */
  if (useInitHeadDirection) {
    HEarth2Earth(headingRad);
  }

  /* Step: 3 to global gps position*/
  sensor_msgs::NavSatFix global_pos;
  dji_osdk_ros::WaypointV2 wpV2;

  for (int i = 0; i < LocalPosVec.size(); ++i) {

    double ref[3], result[3];
    ref[0] = homePosition.latitude;
    ref[1] = homePosition.longitude;
    ref[2] = homePosition.altitude;

    TOOLS::Meter2LatLongAlt(ref, LocalPosVec[i], result);

    wpV2.latitude = result[0];
    wpV2.longitude = result[1];
    wpV2.relativeHeight = LocalPosVec[i].z;

    wpV2Vec.push_back(wpV2);
  }

  return wpV2Vec;
}
