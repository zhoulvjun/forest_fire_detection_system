#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: fastaiunetvideo.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-12
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------


import cv2
import torch
from torch2trt import TRTModule, torch2trt

import sys
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, show_cv_image, draw_mask, cv_to_tesnor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector_trt = TRTModule().to(device)
detector_trt.load_state_dict(torch.load("./final_trt.pth"))
print("loading params from: ~/catkin_ws/src/forest_fire_detection_system/scripts/vision/UnetDetModel/final_trt.pth")

capture = cv2.VideoCapture("../datas/videoplayback.mp4")

while(1):
    ret, frame = capture.read()
    img_ = cv_to_tesnor(frame, 255, 255, device)

    pre = detector_trt(img_)
    cv_mask = tensor_to_cv(pre[0].cpu())
    # show_cv_image(cv_mask,"cv")

    masked_img = draw_mask(cv2.resize(frame, (255,255)), cv_mask)
    show_cv_image(masked_img,"cv")

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
print("end")
