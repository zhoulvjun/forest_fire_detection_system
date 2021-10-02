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

import os
import numpy as np

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

import torch
from UnetDetModel import UnetModel

# The parameters to control the final imgae size
RESIZE_WIDTH = 255
RESIZE_HEIGHT = 255


class FireSmokeDetector(object):

    def __init__(self):

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        # ros stuff
        self.rate = rospy.Rate(1)
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.image_cb)

        # detection model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = UnetModel.pureunet(
            in_channels=3, out_channels=1) .to(self.device)
        self.param_path = os.path.expanduser(
            '~/catkin_ws/src/forest_fire_detection_system/scripts/vision/UnetDetModel/final.pth')
        self.detector.load_state_dict(torch.load(self.param_path))

        # hint
        rospy.loginfo("loading params from: ~/catkin_ws/src/forest_fire_detection_system/scripts/vision/UnetDetModel/final.pth")

    def image_cb(self, msg):

        self.ros_image = msg

        if self.ros_image is None:
            rospy.loginfo("waiting for the image")
        else:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')

    def cv_to_tesnor(self, cv_img, re_width=RESIZE_WIDTH, re_height=RESIZE_HEIGHT):
        """

        Note: The input tensor could be (0.0~255.0), but should be float

        """

        # cv(BGR) --> tensor(RGB)
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # resize the image for tensor
        img = cv2.resize(img, (re_width, re_height))

        # change the shape order and add bathsize
        img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        return img_.to(self.device)

    def tensor_to_cv(self, ten):
        """

        Note: The tensor could be any value, but cv_image should in (0~255)
        <uint8>

        """
        # tensor --> numpy
        np_array = ten.detach().numpy()

        # normalize
        maxValue = np_array.max()
        np_array = (np_array/maxValue)*255
        mat = np.uint8(np_array)

        # change thw dimension shape to fit cv image
        mat = np.transpose(mat, (1, 2, 0))

        return mat

    def show_cv_image(self, mat, title: str):
        cv2.imshow(title, mat)
        cv2.waitKey(3)

    def run(self):

        # for save the original video
        output_org_video = cv2.VideoWriter('org_video.avi', cv2.VideoWriter_fourcc(
            *'DIVX'), 5, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # for save the masked video
        output_masked_video = cv2.VideoWriter('mask_video.avi', cv2.VideoWriter_fourcc(
            *'DIVX'), 5, (RESIZE_WIDTH, RESIZE_HEIGHT))

        while not rospy.is_shutdown():

            if self.cv_image is None:
                rospy.loginfo("Waiting for ros image!")
            else:
                # Step 0: subscribe the image, covert to cv image, and store
                output_org_video.write(self.cv_image)

                # Step 1: convert the cv image to tensor.
                tensor_img = self.cv_to_tesnor(self.cv_image)

                # Step 2: feed tensor to detector
                tensor_mask = self.detector(tensor_img)

                # Step 3: mask to cv image mask
                cv_mask = self.tensor_to_cv(tensor_mask[0].cpu())

                # Step 4: merge the cv_mask and original cv_mask
                cv_final_img = cv2.resize(
                    self.cv_image, (RESIZE_WIDTH, RESIZE_HEIGHT))

                cv_final_img[:, :, 0] = cv_mask[:, :, 0] * \
                    0.5 + cv_final_img[:, :, 0]*0.7

                channel_max = cv_final_img[:, :, 0].max()
                norm_channel = (cv_final_img[:, :, 0]/channel_max)*255
                cv_final_img[:, :, 0] = np.uint8(norm_channel)

                # Step 5: show the mask
                self.show_cv_image(cv_final_img, 'cv_mask')

                # Step 6: save the video.
                output_masked_video.write(cv_final_img)

            self.rate.sleep()

        # end of the saving video
        output_org_video.release()
        output_masked_video.release()

        rospy.loginfo("end of the saving video!")


if __name__ == '__main__':
    rospy.init_node("detection_fire_smoke_node", anonymous=True)
    detector = FireSmokeDetector()
    detector.run()
