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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def save_checkpoint(state, filename = "unetlight_checkpoint.pth"):
    print(" ===> saving checkpoint")
    torch.save(state, filename)

# def load_checkpoint(checkpoint, model):
#     print("===> loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir,
                batch_size, train_transform, val_transform,
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

    # val data set
    val_ds = CarvanaDataset( 
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader, val_loader

def check_accuracy(loader, model, device = "cuda"):
    num_corrrect = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_corrrect += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (( preds + y ).sum() + 1e-8)

    print(
        f" Got {num_corrrect}/{num_pixels} with acc {num_corrrect/num_pixels * 100: .2f}"
        )

    print(f"Dice Score: {dice_score/len(loader)}")

    model.train()

def save_predictions_as_imgs(loader, model, folder = "saved_images/",
                             device = "cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/pred_{idx}.png"
        )

    model.train()

