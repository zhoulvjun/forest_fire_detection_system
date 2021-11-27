/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RGB_IRSeperator.cpp
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

#include <modules/ImgVideoOperator/RGB_IRSeperator.hpp>

void FFDS::MODULES::RGB_IRSeperator::imageCallback(
    const sensor_msgs::Image::ConstPtr& img) {
  rawImgPtr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  rawImg = rawImgPtr->image;
}

void FFDS::MODULES::RGB_IRSeperator::run() {
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/H20T_Camera.yaml";
  PRINT_INFO("get camera params from %s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);

  int irImgWid = FFDS::TOOLS::getParam(node, "pure_IR_width", 960);
  int irImgHet = FFDS::TOOLS::getParam(node, "pure_IR_height", 770);

  int rgbImgWid = FFDS::TOOLS::getParam(node, "pure_RGB_width", 960);
  int rgbImgHet = FFDS::TOOLS::getParam(node, "pure_RGB_height", 770);

  int upperBound = FFDS::TOOLS::getParam(node, "upper_bound", 336);
  int lowerBound = FFDS::TOOLS::getParam(node, "lower_bound", 1106);

  int irUpLeft_x = 0;
  int irUpLeft_y = upperBound;

  int rgbUpLeft_x = irImgWid + 1;
  int rgbUpLeft_y = upperBound;

  while (ros::ok()) {
    ros::spinOnce();
    cv::Mat irImg =
        rawImg(cv::Rect(irUpLeft_x, irUpLeft_y, irImgWid, irImgHet));
    cv::Mat rgbImg =
        rawImg(cv::Rect(rgbUpLeft_x, rgbUpLeft_y, rgbImgWid, rgbImgHet));

    sensor_msgs::ImagePtr irMsg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", irImg).toImageMsg();
    sensor_msgs::ImagePtr rgbMsg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgbImg).toImageMsg();

    imageIRPub.publish(irMsg);
    imageRGBPub.publish(rgbMsg);

    ros::Rate(10).sleep();
  }
}
