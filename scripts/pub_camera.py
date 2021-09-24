#! /usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError

capture = cv2.VideoCapture(0)
capture.set(15,-0.1)
if __name__ == "__main__":
    capture.open(0)
    rospy.init_node('Camera', anonymous=True)  # 定义节点
    image_pub = rospy.Publisher('/camera/rgb/image_raw',
                                Image,
                                queue_size=10)  # 定义话题

    while not rospy.is_shutdown():
        ret, frame = capture.read()
        # frame = cv2.flip(frame, 0)  # 镜像操作
        # frame = cv2.flip(frame, 1)  # 镜像操作

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
