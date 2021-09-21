import torch
import torchvision
import numpy as np

import os
import cv2
import matplotlib.pyplot as plt


# testing te camera
video_session = cv2.VideoCapture(0)

# conver 2 RGB
def grab_frame(cap):
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
ax2.set_title("Mask")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

# Create two image objects to picture on top of the axes defined above
im1 = ax1.imshow(grab_frame(video_session))
im2 = ax2.imshow(grab_frame(video_session))
#%%
plt.ion()
plt.show()

