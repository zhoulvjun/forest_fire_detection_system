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

from forest_fire_detection_system.msg import SingleFirePosIR


class PotentialFireIrFinder(object):

    def __init__(self):

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        self.pot_fire_pos = SingleFirePosIR()
        self.pot_fire_pos.is_pot_fire = False

        # ros stuff
        self.rate = rospy.Rate(5)
        self.image_sub = rospy.Subscriber(
            "dji_osdk_ros/main_camera_images", Image, self.image_cb)
        self.fire_pos_pub = rospy.Publisher(
            "forest_fire_detection_system/single_fire_pos_ir_img", SingleFirePosIR, queue_size=10)

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

        windowSize = [80, 80]
        stepSize = 40

        for (x, y, patch) in self.sliding_window(binary_img, stepSize, windowSize):
            coord_list.append([x, y])
            judje = patch > 0
            judge_list.append(np.count_nonzero(judje))

        if (np.count_nonzero(judge_list) != 0):

            best_index = judge_list.index(max(judge_list))
            best_pos = coord_list[best_index]

            self.pot_fire_pos.is_pot_fire = True
            self.pot_fire_pos.x = best_pos[0]+windowSize[0]/2
            self.pot_fire_pos.y = best_pos[1]+windowSize[1]/2

            rospy.loginfo("pot_fire_pos.x: %d", self.pot_fire_pos.x)
            rospy.loginfo("pot_fire_pos.y: %d", self.pot_fire_pos.y)

            cv2.rectangle(self.cv_image, (best_pos[0], best_pos[1]), (
                best_pos[0] + windowSize[0], best_pos[1] + windowSize[1]), (0, 255, 0), 2)
        else:
            self.pot_fire_pos.x = -1
            self.pot_fire_pos.y = -1
            self.pot_fire_pos.is_pot_fire = False
            rospy.loginfo("no potential fire currently!")

        self.pot_fire_pos.img_width = self.cv_image.shape[1]
        self.pot_fire_pos.img_height = self.cv_image.shape[0]
        self.fire_pos_pub.publish(self.pot_fire_pos)

        # for display
        cv_image = cv2.resize(self.cv_image, (960, 540))
        cv2.imshow("Window", cv_image)
        cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():

            if self.cv_image is not None:

                _, binary = cv2.threshold(
                    self.cv_image[:, :, 2], 25, 255, cv2.THRESH_BINARY)

                # opening operation
                kernel = np.ones((2, 2), np.uint8)
                opening = cv2.morphologyEx(
                    binary, cv2.MORPH_OPEN, kernel, iterations=2)

                self.find(opening)

            else:
                rospy.loginfo("waiting for the image")

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node("find_potential_fire_ir_node", anonymous=True)
    detector = PotentialFireIrFinder()
    detector.run()
