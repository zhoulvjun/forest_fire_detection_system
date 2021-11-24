/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: AlianRGB_IR.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-23
 *
 *   @Description:
 *
 *******************************************************************************/
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

void showColorGrayView(const sensor_msgs::ImageConstPtr msgImg) {
  cv_bridge::CvImagePtr cvImgPtr;
  try {
    cvImgPtr = cv_bridge::toCvCopy(msgImg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception e) {
    ROS_ERROR_STREAM("Cv_bridge Exception:" << e.what());
    return;
  }
  cv::Mat cvColorImgMat = cvImgPtr->image;
  cv::Mat cvGrayImgMat;
  cv::cvtColor(cvColorImgMat, cvGrayImgMat, CV_BGR2GRAY);
  cv::imshow("colorview", cvColorImgMat);
  cv::imshow("grayview", cvGrayImgMat);
  cv::waitKey(5);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "grayview");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  cv::namedWindow("colorview", cv::WINDOW_NORMAL);
  cv::moveWindow("colorview", 100, 100);
  cv::namedWindow("grayview", cv::WINDOW_NORMAL);
  cv::moveWindow("grayview", 600, 100);
  image_transport::Subscriber sub =
      it.subscribe("/rgb/image_raw", 1, showColorGrayView);
  ros::spin();

  return 0;
}
