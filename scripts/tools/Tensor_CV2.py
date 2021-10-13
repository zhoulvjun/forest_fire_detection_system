#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
#
#   @Filename: Tensor_CV2.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-20
#
#   @Email: 2015097272@qq.com
#
#   @Description: some functions between the opencv and pytorch
#
# ------------------------------------------------------------------------------

import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch


def cv_to_tesnor(cv_img, re_width, re_height, device):
    """

    Description: This function convert "BGR" image --> tensor(1, 3, H, W)

    Note: The input tensor could be (0.0~255.0), but should be float

    """

    # cv(BGR) --> tensor(RGB)
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # resize the image for tensor
    img = cv2.resize(img, (re_width, re_height))

    # change the shape order and add bathsize
    img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

    return img_.to(device)


def tensor_to_cv(ten):
    """

    Note: The tensor could be any value, but cv_image should in (0~255)
    <uint8>

    """
    # tensor --> numpy
    np_array = ten.detach().numpy()

    # normalize
    maxValue = np_array.max()
    np_array = (np_array/maxValue)*255
    mat = np.uint8(np_array)

    # change thw dimension shape to fit cv image
    mat = np.transpose(mat, (1, 2, 0))

    return mat


def draw_mask(cv_org_img, cv_mask):

    cv_org_img[:, :, 1] = cv_mask[:, :, 0] * \
        0.8 + cv_org_img[:, :, 1]*0.5

    channel_max = cv_org_img[:, :, 1].max()
    norm_channel = (cv_org_img[:, :, 1]/channel_max)*255
    cv_org_img[:, :, 1] = np.uint8(norm_channel)

    return cv_org_img
