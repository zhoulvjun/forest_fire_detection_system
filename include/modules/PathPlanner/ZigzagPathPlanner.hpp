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

#include <dji_osdk_ros/WaypointV2.h>
#include <modules/PathPlanner/PathPlannerBase.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>

#include <common/CommonTypes.hpp>
#include <tools/MathLib.hpp>

namespace FFDS {
namespace MODULES {

class ZigzagPathPlanner : public MODULES::PathPlannerBase {

public:
  ZigzagPathPlanner(){};
  ZigzagPathPlanner(sensor_msgs::NavSatFix home, int num, float len, float wid,
                    float height)
      : homeGPos(home), zigzagNum(num), zigzagLen(len), zigzagWid(wid),
        zigzagHeight(height){};

  ~ZigzagPathPlanner(){};

  void setParams(sensor_msgs::NavSatFix home, int num, float len, float wid,
                 float height);

  std::vector<dji_osdk_ros::WaypointV2> &
  getWpV2Vec(bool isGlobal, bool useInitHeadDirection, float homeHeadRad);

private:
  int zigzagNum{0};
  float zigzagLen{0.0};
  float zigzagWid{0.0};
  float zigzagHeight{0.0};
  sensor_msgs::NavSatFix homeGPos;

  std::vector<COMMON::LocalPosition> LocalPosVec;
  std::vector<dji_osdk_ros::WaypointV2> wpV2Vec;

  /**
   * NOTE: we want the M300 initial heading as the positive direction.
   * NOTE: This coordinates is defined as H(ead)Earth coordinates by Shun.
   * NOTE: There is only one difference between HEarth and Earth, the
   * NOTE: init-heading angle.
   **/

  /* generate the local as the same, treat it in HEarth or Earth local position.
   */
  void calLocalPos();

  void HEarth2Earth(float homeHeadRad);

  void feedWp2Vec(bool isGlobal);
};

} // namespace MODULES
} // namespace FFDS

#endif /* ZIGZAGPATHPLANNER_HPP */
