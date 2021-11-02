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

    def image_seg(self,gray, threshold=20):

        ret, binary = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY) 

        # opening
        kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel, iterations = 2)

        # 将图像等分小块，然后找含有最多的白点的块作为相机中心点。。。:


        cv2.imshow("img", self.cv_image)  
        cv2.waitKey(0)


    def run(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                self.image_seg(self.cv_image[:,:,2])


if __name__ == '__main__':
    rospy.init_node("thermal_detector_node", anonymous=True)
    detector = ThermalDetector()
    detector.run()
