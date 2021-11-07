#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: seperate_original_image.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-07
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------

import cv2
from cv_bridge import CvBridge
from forest_fire_detection_system.msg import SingleFirePosIR
import numpy as np
import rospy
from sensor_msgs.msg import Image
import yaml
from yaml import CLoader


class OriginalImageSeperator(object):
    def __init__(self):
        # read the camera parameters
        config_path = open(
            "/home/shun/catkin_ws/src/forest_fire_detection_system/config/H20T_IR_Camera.yaml"
        )
        self.H20T = yaml.load(config_path, Loader=CLoader)

        self.full_img = np.zeros(
            (self.H20T["full_img_height"], self.H20T["full_img_width"], 3),
            dtype='uint8')
        self.pure_ir_img = np.zeros(
            (self.H20T["pure_IR_height"], self.H20T["pure_IR_width"], 3),
            dtype='uint8')
        self.pure_rgb_img = np.zeros(
            (self.H20T["pure_RGB_height"], self.H20T["pure_RGB_width"], 3),
            dtype='uint8')

        self.ros_image = Image()
        self.convertor = CvBridge()

        rospy.wait_for_message("dji_osdk_ros/main_camera_images", Image)
        self.image_sub = rospy.Subscriber("dji_osdk_ros/main_camera_images",
                                          Image, self.image_cb)

    def image_cb(self, msg):
        self.ros_image = msg
        self.full_img = self.convertor.imgmsg_to_cv2(self.ros_image, 'bgr8')

        # 1920 x 1440
        # rospy.loginfo("ros Image size(W x H): %d x %d", self.ros_image.width,
        #         self.ros_image.height)
        # rospy.loginfo("cv Image size(W x H): %d x %d", full_img.shape[1],
        #         full_img.shape[0])

        self.pure_ir_img = self.full_img[
            self.H20T["upper_bound"]:self.H20T["lower_bound"], :self.
            H20T["pure_IR_width"], :]

        print(self.pure_ir_img.shape)
        cv2.imshow("ir", self.pure_ir_img)

        self.pure_rgb_img = self.full_img[
            self.H20T["upper_bound"]:self.H20T["lower_bound"],
            self.H20T["pure_RGB_width"]:, :]

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("seperate_original_image_node", anonymous=True)
    detector =OriginalImageSeperator()
    detector.run()
