#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: pub_local_video.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-13
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------


import os
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image


if __name__ == "__main__":

    video_name = os.path.expanduser("~/catkin_ws/src/forest_fire_detection_system/scripts/develop/datas/DJI_0261.MP4")
    capture = cv2.VideoCapture(video_name)
    bridge = CvBridge()

    rospy.init_node('Camera', anonymous=True)
    image_pub = rospy.Publisher(
        "dji_osdk_ros/main_camera_images", Image, queue_size=10)

    while not rospy.is_shutdown():

        ret, frame = capture.read()
        ros_img = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_pub.publish(ros_img)

        # rospy.loginfo("video from:", video_name)

        rate = rospy.Rate(10)

    capture.release()
    cv2.destroyAllWindows()
