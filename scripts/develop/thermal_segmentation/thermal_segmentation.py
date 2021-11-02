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


class ThermalDetector(object):

    def __init__(self):

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = cv2.imread(os.path.expanduser('~/1.png'))

        # ros stuff
        self.rate = rospy.Rate(5)
        # self.image_sub = rospy.Subscriber(
        #     "dji_osdk_ros/main_camera_images", Image, self.image_cb)

    def image_cb(self, msg):

        self.ros_image = msg

        if self.ros_image is None:
            rospy.loginfo("waiting for the image")
        else:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')

    def gen_mesh(self, width, height, mesh_size):

        int_wid_num = width//mesh_size
        int_hei_num = height//mesh_size
        wid_step = [mesh_size for i in range(int_wid_num)]
        hei_step = [mesh_size for i in range(int_hei_num)]

        wid_step.append(width % mesh_size)
        hei_step.append(height % mesh_size)

        return wid_step, hei_step

    def image_seg(self, gray, threshold=20):

        ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # opening
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        wid_step_list, hei_step_list = self.gen_mesh(
            self.cv_image.shape[1], self.cv_image.shape[0], 20)

        # print(self.cv_image.shape, wid_step, hei_step)

        fore_hei = 0
        back_hei = 0
        fore_wid = 0
        back_wid = 0

        coord_list = []
        judge_list= []

        for wid_step in wid_step_list:
            fore_wid += wid_step
            for hei_step in hei_step_list:
                fore_hei += hei_step

                cur_square = opening[back_hei:fore_hei+1, back_wid:
                        fore_wid+1]
                coord_list.append([fore_hei,back_hei, fore_wid, back_wid])
                judge_square = cur_square>0
                judge_list.append(np.count_nonzero(judge_square))

                back_hei = fore_hei

            fore_hei = 0
            back_hei = 0

            back_wid = fore_wid


        best_index = judge_list.index(max(judge_list))
        best_pos = coord_list[best_index]
        print(best_pos)

        cv2.rectangle(self.cv_image, (best_pos[3], best_pos[1]), (best_pos[2],
            best_pos[0]), ( 255, 255, 255), 1)

        cv2.imshow("img", self.cv_image)
        cv2.waitKey(0)
        cv2.imshow("img", opening)
        cv2.waitKey(0)

    def run(self):
        # while not rospy.is_shutdown():
        if self.cv_image is not None:
            self.image_seg(self.cv_image[:, :, 2])


if __name__ == '__main__':
    rospy.init_node("thermal_detector_node", anonymous=True)
    detector = ThermalDetector()
    detector.run()
