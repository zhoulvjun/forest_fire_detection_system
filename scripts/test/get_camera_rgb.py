#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Lee Ltd. All rights reserved.
#
#   @Filename: get_camera_rgb.py
#
#   @Author: lee-shun
#
#   @Date: 2021-09-24
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

# FIXME: There is a bug when calling the severices.

import rospy
from dji_osdk_ros.srv import SetupCameraStream
from sensor_msgs.msg import Image


class GetImageNode(object):
    def __init__(self):
        self.image_frame = Image()
        self.rate = rospy.Rate(10)

        rospy.wait_for_service("setup_camera_stream")
        self.set_camera_cli = rospy.ServiceProxy("setup_camera_stream",
                                                 SetupCameraStream)

        rospy.Subscriber("dji_osdk_ros/fpv_camera_images",
                         Image, self.image_cb)

        self.image_pub = rospy.Publisher(
            '/camera/rgb/image_raw', Image, queue_size=10)

    def image_cb(self, msg):
        self.image_frame = msg

    def run(self):

        set_camera_handle = SetupCameraStream()
        result = self.set_camera_cli(0, 1)
        print("the result is", result)

        while not rospy.is_shutdown():
            self.image_pub.publish(self.image_frame)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('get_image_node', anonymous=True)

    node = GetImageNode()
    node.run()
