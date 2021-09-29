#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Lee Ltd. All rights reserved.
#
#   @Filename: detection.py
#
#   @Author: Qiao Linhan
#
#   @Date: 2021-09-20
#
#   @Email: 2015097272@qq.com
#
#   @Description: describe the camera stream, detect and mask, save the video
#
# ------------------------------------------------------------------------------

# TODO: 1. check if the fastai can be installed on jetson;
# TODO: 2. the cv_bridge is good with ros-opencv and opencv4?


import rospy
import torch
import torchvision
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class FireSmokeDetector(object):
    def __init__(self):
        self.rate = rospy.Rate(1)
        self.convertor = CvBridge()

        # ros topic
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_cb)

    def image_cb(self, msg):
        self.ros_image = msg
        self.cv_image = self.convertor.imgmsg_to_cv2(
            img_msg=self.ros_image, desired_encoding='passthrough')

    def load_model(self):
        pass

    def write_result(self):
        pass

    def show_image_info(self):
        pass

    def run(self):
        while not rospy.is_shutdown():
            self.show_image_info()
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node("fire_smoke_detecting_node", anonymous=False)
