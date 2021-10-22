#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: GenerateTrt.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-29
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import torch
from UnetModel import pureunet
from torch2trt import torch2trt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = pureunet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x])


torch.save(detector_trt.state_dict(), 'final_trt.pth')
