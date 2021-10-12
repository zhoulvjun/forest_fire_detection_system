#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
#
#   @Filename: detection.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-20
#
#   @Email: 2015097272@qq.com
#
#   @Description: describe the camera stream, detect and mask, save the video
#
# ------------------------------------------------------------------------------

from tools.Tensor_CV2 import cv_to_tesnor, tensor_to_cv, draw_mask
import os

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import torch
from torch2trt import TRTModule

import sys
PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH+'scripts/')


# The parameters to control the final imgae size
BATCH_SIZE = 1
RESIZE_WIDTH = 255
RESIZE_HEIGHT = 255


class FireSmokeDetector(object):

    def __init__(self):

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        # ros stuff
        self.rate = rospy.Rate(5)
        self.image_sub = rospy.Subscriber(
            "dji_osdk_ros/main_camera_images", Image, self.image_cb)

        # detection model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.param_path = os.path.expanduser(
            '~/catkin_ws/src/forest_fire_detection_system/scripts/vision/UnetDetModel/final_trt.pth')

        self.detector_trt = TRTModule().to(self.device)
        self.detector_trt.load_state_dict(torch.load(self.param_path))

        rospy.loginfo(
            "loading params from: ~/catkin_ws/src/forest_fire_detection_system/scripts/vision/UnetDetModel/final_trt.pth")

    def image_cb(self, msg):

        self.ros_image = msg

        if self.ros_image is None:
            rospy.loginfo("waiting for the image")
        else:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')

    def run(self):

        # for save the masked video
        output_org_video = cv2.VideoWriter('org_video.avi', cv2.VideoWriter_fourcc(
            *'DIVX'), 5, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # for save the masked video
        output_masked_video = cv2.VideoWriter('mask_video.avi', cv2.VideoWriter_fourcc(
            *'DIVX'), 5, (RESIZE_WIDTH, RESIZE_HEIGHT))

        while not rospy.is_shutdown():

            if self.cv_image is None:
                rospy.loginfo("Waiting for ros image!")
            else:
                # STEP: 0 subscribe the image, covert to cv image.

                # STEP: 1 convert the cv image to tensor.
                tensor_img = cv_to_tesnor(
                    self.cv_image, RESIZE_WIDTH, RESIZE_HEIGHT, self.device)

                # STEP: 2 feed tensor to detector
                tensor_mask = self.detector_trt(tensor_img)

                # STEP: 3 mask to cv image mask
                cv_mask = tensor_to_cv(tensor_mask[0].cpu())

                # STEP: 4 merge the cv_mask and original cv_mask
                cv_org_img = cv2.resize(
                    self.cv_image, (RESIZE_WIDTH, RESIZE_HEIGHT))

                # save before merge
                output_org_video.write(cv_org_img)

                cv_final_img = draw_mask(cv_org_img, cv_mask)

                # STEP: 5 show the mask
                cv2.imshow('cv_mask', cv_final_img)
                cv2.waitKey(3)

                # STEP: 6 save the video.
                output_masked_video.write(cv_final_img)

            self.rate.sleep()

        # end of the saving video
        output_masked_video.release()

        rospy.loginfo("end of the saving masked video!")


if __name__ == '__main__':
    rospy.init_node("detection_fire_smoke_node", anonymous=True)
    detector = FireSmokeDetector()
    detector.run()
