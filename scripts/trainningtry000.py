#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Lee Ltd. All rights reserved.
#
#   @Filename: detection.py
#
#   @Author: Qiao Linhan
#
#   @Date: 2021-09-20
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from try000 import UNET

# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_prediciton_as_imgs,
# )

# hyper-parameters etc.
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_epochs = 3
num_workers = 2
image_height = 160 # original 1280
image_width = 240 # original 1918
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_images'
TRAIN_MASK_DIR = 'data/train_masks'
VAL_IMG_DIR = 'data/val_images'
VAL_MASK_DIR = 'data/val_masks'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = device)
        targets = targets.float().unsqueeze(1).to(device = device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqam loop
        loop.set_postfix(loss = loss.item()) # for 1 epoch
        


def main():
    pass

if __name__ == '__main__':
    main()
