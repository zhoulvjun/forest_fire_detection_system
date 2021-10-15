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
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image


if __name__ == "__main__":

    rospy.init_node('Camera', anonymous=True)

    # video_name = os.path.expanduser("~/DJI_0026.MOV")
    video_name = os.path.expanduser("~/videoplayback.mp4")
    capture = cv2.VideoCapture(video_name)
    bridge = CvBridge()

    rospy.loginfo("video from: "+ video_name)

    image_pub = rospy.Publisher(
        "dji_osdk_ros/main_camera_images", Image, queue_size=10)

    while not rospy.is_shutdown():
        ret, frame = capture.read()

        if frame is not None:
            cv2.imshow("frame",frame)
            cv2.waitKey(3)
            ros_img = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_pub.publish(ros_img)
        else:
            cv2.destroyAllWindows()
            rospy.loginfo("None frame! end of video")
            break
        rate = rospy.Rate(2)

    capture.release()
