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
#   @Date: 2021-09-29
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------
from albumentations.pytorch.transforms import ToTensor
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from fastaiunetmodel000 import pureunet

from fastaiunetutils000 import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# hyper-parameters etc.
LEARNING_RATE = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1 # 32
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 255 # original 1280
IMAGE_WIDTH = 255 # original 1918
PIN_MEMORY = True
# LOAD_MODEL = False # original
LOAD_MODEL = False
TRAIN_IMG_DIR = 'datas/fs'
TRAIN_MASK_DIR = 'datas/fslabel'
VAL_IMG_DIR = 'datas/fs'
VAL_MASK_DIR = 'datas/fslabel'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE) # float, cross-entropy
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqam loop show loss
        loop.set_postfix(loss = loss.item()) # for 1 epoch


# data augmentation
def main():
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 35, p = 1.0), # rotation
            A.HorizontalFlip(p = 0.5), # flip
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0 # deviding by 255
            ),
            ToTensorV2()
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 35, p = 1.0), # rotation
            A.HorizontalFlip(p = 0.5), # flip
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0 # deviding by 255
            ),
            ToTensorV2()
        ]
    )

    model = pureunet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # binary cross-entropy with logits
    # loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # data_loader
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("pureunet.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model 
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        #ßsave_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder = "forest_fire_detection_system/scripts/saved_images/", device = DEVICE
        # )

    torch.save(model.state_dict(), 'final.pth')

if __name__ == '__main__':
    main()
