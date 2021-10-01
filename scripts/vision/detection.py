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

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image

from UnetDetModel import UnetModel
import torch


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
        self.dic_path = '/home/ls/catkin_ws/src/forest_fire_detection_system/scripts/vision/final.pth'
        self.detector.load_state_dict(torch.load(self.dic_path))

    def image_cb(self, msg):
        self.ros_image = msg

        if self.ros_image is not None:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')
        else:
            rospy.loginfo("waiting for the image")

    def cv_to_tesnor(self, cv_img, re_width=255, re_height=255):
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
        mat = np_array/maxValue  # (0~1)
        # np_array = np_array*255/maxValue
        # mat = np.uint8(np_array)

        # change thw dimension shape to fit cv image
        mat = np.transpose(mat, (1, 2, 0))

        return mat

    def feed_img_2_model(self):
        img_ = self.cv_to_tesnor(self.cv_image)
        self.model_result = self.detector(img_)

    def show_cv_image(self, mat, title: str):
        cv2.imshow(title, mat)
        cv2.waitKey(3)

    def write_cv_file(self, cv_image):

        frameSize = (500, 500)
        wirte = cv2.VideoWriter(
            'output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
        for filename in glob.glob('D:/images/*.jpg'):
            img = cv2.imread(filename)
            out.write(img)
        out.release()

    def run(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                self.feed_img_2_model()
                mat = self.tensor_to_cv(self.model_result[0].cpu())
                self.show_cv_image(mat, 'result')

            else:
                rospy.loginfo("waiting for ros image!")
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node("fire_smoke_detecting_node", anonymous=True)
    detector = FireSmokeDetector()
    detector.run()
