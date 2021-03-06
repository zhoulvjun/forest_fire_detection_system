#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: TestPIL.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-10
#
#   @Email: 2015097272@qq.com
#
#   @Description: The usage to PIL
#
#------------------------------------------------------------------------------


from PIL import Image
import numpy as np
import cv2

im = Image.open('./datas/Smoke_segmentation/testing/image_00000.jpg')
im_seg = Image.open('./datas/Smoke_segmentation/gt_testing/image_00000.jpg')

mask = np.array(im_seg.convert("L"), dtype=np.float32)
mask = mask/255.0
mask[mask>0.0] =1.0

print(mask)
cv2.imshow('hh',mask)
cv2.waitKey(0)
