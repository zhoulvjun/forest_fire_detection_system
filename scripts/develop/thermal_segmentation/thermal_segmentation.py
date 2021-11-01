#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: thermal_segmentation.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-01
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import os
import numpy as np

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import sys
PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH+'scripts/')


class ThermalDetector(object):

    def __init__(self):

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        # ros stuff
        self.rate = rospy.Rate(5)
        self.image_sub = rospy.Subscriber(
            "dji_osdk_ros/main_camera_images", Image, self.image_cb)

    def image_cb(self, msg):

        self.ros_image = msg

        if self.ros_image is None:
            rospy.loginfo("waiting for the image")
        else:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')

    def run(self):
        while not rospy.is_shutdown():

            if self.cv_image is not None:
                red = self.cv_image[:,:,2]
                red[red>=30]=255
                red[red<30]=0

                fire_pixel_index = np.argwhere(red==255)
                cg = fire_pixel_index.mean(axis=0).astype(np.uint8)
                print(cg.shape)
                cv2.circle(self.cv_image, (cg[0],cg[1]), 10,(0,0,255))
                cv2.imshow('red',red)
                cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node("thermal_detector_node", anonymous=True)
    detector = ThermalDetector()
    detector.run()
