#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: find_potential_fire_IR.py
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

import numpy as np

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge


class PotentialFireIrFinder(object):

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

    def sliding_window(self, image, stepSize=10, windowSize=[20, 20]):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def find(self, binary_img):

        judge_list = []
        coord_list = []

        for (x, y, window) in self.sliding_window(binary_img, 10, [20, 20]):
            patch = binary_img[y:y+21, x:x+21]
            coord_list.append([x, y])
            judje = patch > 0
            judge_list.append(np.count_nonzero(judje))

        if (np.count_nonzero(judge_list) != 0):
            best_index = judge_list.index(max(judge_list))
            best_pos = coord_list[best_index]

            clone = self.cv_image.copy()
            cv2.rectangle(clone, (best_pos[0], best_pos[1]),
                          (best_pos[0] + 21, best_pos[1] + 21), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():

            if self.cv_image is not None:
                rospy.loginfo("waiting for the image")
                _, binary = cv2.threshold(
                    self.cv_image[:, :, 2], 25, 255, cv2.THRESH_BINARY)

                # opening
                kernel = np.ones((2, 2), np.uint8)
                opening = cv2.morphologyEx(
                    binary, cv2.MORPH_OPEN, kernel, iterations=2)

                self.find(opening)


if __name__ == '__main__':
    rospy.init_node("find_potential_fire_ir_node", anonymous=True)
    detector = PotentialFireIrFinder()
    detector.run()

