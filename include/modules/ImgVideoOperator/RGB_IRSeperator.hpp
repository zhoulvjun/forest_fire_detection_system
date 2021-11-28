/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RGB_IRSeperator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-27
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_
#define INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <tools/SystemLib.hpp>

namespace FFDS {
namespace MODULES {
class RGB_IRSeperator {
 public:
  RGB_IRSeperator() {
    imageSub = nh.subscribe("dji_osdk_ros/main_camera_images", 10,
                            &RGB_IRSeperator::imageCallback, this);
    imageIRPub =
        it.advertise("forest_fire_detection_system/main_camera_ir_image", 1);
    imageRGBPub =
        it.advertise("forest_fire_detection_system/main_camera_rgb_image", 1);

    ros::Duration(3.0).sleep();
  }

  void run();

 private:
  ros::NodeHandle nh;
  image_transport::ImageTransport it{nh};

  ros::Subscriber imageSub;

  image_transport::Publisher imageIRPub;
  image_transport::Publisher imageRGBPub;

  cv_bridge::CvImagePtr rawImgPtr;
  cv::Mat rawImg;

  void imageCallback(const sensor_msgs::Image::ConstPtr& img);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_
