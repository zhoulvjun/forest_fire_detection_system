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
# import torch
# import torchvision
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# TODO: import the model(qiao)


class FireSmokeDetector(object):
    def __init__(self):
        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        self.rate = rospy.Rate(1)
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.image_cb)

    def image_cb(self, msg):
        self.ros_image = msg

        if self.ros_image is not None:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')
        else:
            rospy.loginfo("waiting for the image")

    def image_cb2(self, msg):
        self.ros_image = msg

        if self.ros_image is not None:
            # TODO: without cv_bridge?
            pass
        else:
            rospy.loginfo("waiting for the image")
        pass

    def load_model(self):
        pass

    def write_result(self):
        pass

    def show_image_info(self, title: str):
        if self.cv_image is None:
            rospy.loginfo("no ros Image to show!")
        else:
            cv2.imshow(title, self.cv_image)
            cv2.waitKey(3)

    def run(self):
        while not rospy.is_shutdown():
            self.show_image_info("cv image from ros-Image")
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node("fire_smoke_detecting_node", anonymous=True)
    detector = FireSmokeDetector()
    detector.run()
