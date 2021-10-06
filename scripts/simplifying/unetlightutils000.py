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
#   @Date: 2021-09-28
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------

import torch
import torchvision
from unetlightdata000 import loadDataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, dataset
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename = "unetlight_checkpoint.pth"):
    print(" ===> saving checkpoint")
    torch.save(state, filename)
    
def check_performance(loader, model, device = DEVICE):
    num_corrrect = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    # for segmentation, dice score might be more important than the accuracy
    with torch.no_grad(): 
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_corrrect += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (( preds + y ).sum() + 1e-8)
    # accuracy
    print(f" Got {num_corrrect}/{num_pixels} with acc {num_corrrect/num_pixels * 100: .2f}")
    # dice score
    print(f"Dice Score: {dice_score/len(loader)}")

    model.train()

def get_loaders(train_dir, train_maskdir,
                batch_size, train_transform,
                num_workers = 4, pin_memory = True):

    # train data set 
    train_ds = loadDataset( 
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader