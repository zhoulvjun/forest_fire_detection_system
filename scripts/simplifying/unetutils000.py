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

trainset, valset = train_test_split(loadDataset, random_state = 0.2, shuffle = False) # original, shuffle = True

trainset = DataLoader(trainset, batch_size=1, shuffle = False, sampler = None)
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)       

