/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ZigzagPathPlanner.hpp
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

#ifndef __ZIGZAGPATHPLANNER_HPP__
#define __ZIGZAGPATHPLANNER_HPP__

#include <common/PathPlannerBase.hpp>
#include <dji_osdk_ros/WaypointV2.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>

#include <tools/MathLib.hpp>
#include <common/CommonTypes.hpp>

namespace FFDS {

class ZigzagPathPlanner : public COMMON::PathPlannerBase {

public:
  ZigzagPathPlanner(sensor_msgs::NavSatFix home, int num, float len, float wid,
                    float height)
      : homePosition(home), zigzagNum(num), zigzagLen(len), zigzagWid(wid),
        zigzagHeight(height){};

  ~ZigzagPathPlanner();

  std::vector<dji_osdk_ros::WaypointV2> &getGPos(bool useInitHeadDirection,
                                                 float heading);

private:
  int zigzagNum{0};
  float zigzagLen{0.0};
  float zigzagWid{0.0};
  float zigzagHeight{0.0};
  sensor_msgs::NavSatFix homePosition;

  std::vector<dji_osdk_ros::WaypointV2> wpV2Vec;
  std::vector<COMMON::LocalPosition> LocalPosVec;

  /**
   * NOTE: we want the M300 initial heading as the positive direction.
   * NOTE: This coordinates is defined as H(ead)Earth coordinates by Shun.
   * NOTE: There is only one difference between HEarth and Earth, the
   * NOTE: init-heading angle.
   **/

  /* generate the local as the same, treat it in HEarth or Earth local position.
   */
  void calLocalPos();

  void HEarth2Earth(float heading);

  sensor_msgs::NavSatFix local2Global(COMMON::LocalPosition local);
};
} // namespace FFDS

#endif /* ZIGZAGPATHPLANNER_HPP */
