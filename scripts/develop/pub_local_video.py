#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------


import cv2
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import rospy

capture = cv2.VideoCapture("../datas/DJI_0261.MP4")
# capture.set(15, -0.1)

if __name__ == "__main__":

    rospy.init_node('Camera', anonymous=True)  # 定义节点
    image_pub = rospy.Publisher("dji_osdk_ros/main_camera_images",
                                Image,
                                queue_size=10)  # 定义话题

    while not rospy.is_shutdown():
        ret, frame = capture.read()
        c_b, c_g, c_r = cv2.split(frame)
        frame = cv2.merge([c_r, c_g, c_b])
        ros_frame = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = "Camera"
        ros_frame.header = header
        ros_frame.width = 640
        ros_frame.height = 480
        ros_frame.encoding = "rgb8"
        ros_frame.step = 1920
        ros_frame.data = np.array(frame).tobytes()  # 图片格式转换
        image_pub.publish(ros_frame)  # 发布消息

        rate = rospy.Rate(10)  # 10hz

    capture.release()
    cv2.destroyAllWindows()
