#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
#
#   @Filename: detection.py
#
#   @Author: Qiao Linhan
#
#   @Date: 2021-09-29
#
#   @Email: 742954173@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):  # by default, no transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # list all files in folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))  # might RGBA
        # mask_path is going to be 4classes and separate by colors in RGB
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # to see the mask, where it equals to 255
        mask[mask >= 254.0] = 1.0  # going to use sigmoid

        if self.transform is not None:
            # data augmentation
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

