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

namespace FFDS {

class ZigzagPathPlanner : public COMMOM::PathPlannerBase {

public:

  ZigzagPathPlanner(sensor_msgs::NavSatFix home, int num, float len, float wid)
      : homePosition(home), zigzagNum(num), zigzagLen(len), zigzagWid(wid){};

  ~ZigzagPathPlanner();

  /* Local earth-fixed coordinates */
  struct LocalPosition {
    float x{0.0};
    float y{0.0};
    float z{0.0};
  };

private:
  int zigzagNum{0};
  float zigzagLen{0.0};
  float zigzagWid{0.0};
  sensor_msgs::NavSatFix homePosition;

  std::vector<dji_osdk_ros::WaypointV2> wpVec;
  std::vector<LocalPosition> LocalPosVec;


  /**
   * NOTE: we want the M300 initial heading as the positive direction.
   * NOTE: This coordinates is defined as H(ead)Earth coordinates by Shun.
   * NOTE: There is only one difference between HEarth and Earth, the
   * NOTE: init-heading angle.
   **/

  /* generate the local as the same, treat it in HEarth or Earth local position. */
  void calLocalPos();

  void HEarth2Earth();

  void local2Global();

  std::vector<dji_osdk_ros::WaypointV2> getGPos_Earth();

  std::vector<dji_osdk_ros::WaypointV2> getGPos_HEarth();

};
} // namespace FFDS

#endif /* ZIGZAGPATHPLANNER_HPP */
