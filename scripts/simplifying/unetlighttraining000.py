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
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unetlight000 import unetlight
from unetlightutils000 import save_checkpoint, get_loaders, check_performance

# hyper-parameters etc.
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1 # 32
NUM_EPOCHS = 5
NUM_WORKERS = 4
IMAGE_HEIGHT = 255 # original 1280
IMAGE_WIDTH = 255 # original 1918
PIN_MEMORY = True

TRAIN_IMG_DIR = 'datas'
TRAIN_MASK_DIR = 'datas'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE) 
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

    model = unetlight(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # binary cross-entropy with logits
    # loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # data_loader
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        save_checkpoint()
        check_performance(train_loader, model, device=DEVICE)

    torch.save(model.state_dict(),'unetlighttry.pth')


# if __name__ == '__main__':
#     main()
