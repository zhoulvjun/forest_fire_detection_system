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
from torch2trt import TRTModule

import sys
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, draw_mask, cv_to_tesnor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector_trt = TRTModule().to(device)
detector_trt.load_state_dict(torch.load("./final_trt.pth"))
print("loading params from: final_trt.pth")

capture = cv2.VideoCapture("../datas/videoplayback.mp4")

val_transforms = A.Compose(
    [
        A.Resize(height=255, width=255),
        A.Normalize(),
        ToTensorV2(),
    ],
)

while(1):
    ret, frame = capture.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    augmentations = val_transforms(image=img_rgb)
    img_ = augmentations['image']
    img_ = img_.float().unsqueeze(0).to(device=device)

    with torch.no_grad():
        preds = torch.sigmoid(detector_trt(img_))
        preds = (preds > 0.4)

    cv_mask = tensor_to_cv(preds[0].cpu())

    masked_img = draw_mask(cv2.resize(frame, (255,255)), cv_mask)
    cv2.imshow("mask",masked_img)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
print("end")

