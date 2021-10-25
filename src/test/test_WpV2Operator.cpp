/*******************************************************************************
*
*   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
*
*   @Filename: test_WpV2Operator.cpp
*
*   @Author: Shun Li
*
*   @Email: 2015097272@qq.com
*
*   @Date: 2021-10-25
*
*   @Description: rewirte the dji_osdk_ros/sample/waypointV2_node.cpp
*
******************************************************************************/
#include<common/WpV2Operator.hpp>

/**
 * global variable
 * */
sensor_msgs::NavSatFix gps_position_;

void gpsPositionSubCallback(const sensor_msgs::NavSatFix::ConstPtr& gpsPosition)
{
  gps_position_ = *gpsPosition;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_WpV2Operator_node");
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub = nh.subscribe("dji_osdk_ros/gps_position", 10, &gpsPositionSubCallback);
  auto obtain_ctrl_authority_client = nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
    "obtain_release_control_authority");
  
  //if you want to fly without rc ,you need to obtain ctrl authority.Or it will enter rc lost.
  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  obtainCtrlAuthority.request.enable_obtain = true;
  obtain_ctrl_authority_client.call(obtainCtrlAuthority);

  ros::Duration(1).sleep();
  ros::AsyncSpinner spinner(1);
  spinner.start();
  /* runWaypointV2Mission(nh); */

  ros::waitForShutdown();
}

