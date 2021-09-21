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
#   @Email: 742954173@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, self.iamges[index])
        mask_path = os.path.join(self.mask_dir, self.iamges[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32) # mask_path is going to be gray scale, for PIL, L
        # to see the mask, where it equals to 255
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            # data augmentation
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask