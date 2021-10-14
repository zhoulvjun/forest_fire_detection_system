import cv2
import sys
import numpy as np
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, draw_mask
import torch
from PIL import Image
from model import UNET  # seg 4 classes

from torch2trt import torch2trt

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

val_transforms = A.Compose(
    [
        A.Resize(height=255, width=255),
        A.Normalize(),
        ToTensorV2(),
    ],
)

img_rgb = np.array(Image.open("../datas/Smoke_segmentation/training/image_00001.jpg"))

img_cv = cv2.imread("../datas/Smoke_segmentation/training/image_00001.jpg")

augmentations = val_transforms(image=img_rgb)
img_ = augmentations['image']
img_ = img_.float().unsqueeze(0).to(DEVICE)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

with torch.no_grad():
    preds = torch.sigmoid(model(img_))
    preds = (preds > 0.3)

# original model
cv_mask = tensor_to_cv(preds[0].cpu())
cv2.imshow("cv", cv_mask)
print(cv_mask[200,125])
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

# optimied model with thesorrt
init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x], fp16_mode=True)

with torch.no_grad():
    preds = torch.sigmoid(detector_trt(img_))
    preds = (preds > 0.4)

cv_mask = tensor_to_cv(preds[0].cpu())
cv2.imshow("cv", cv_mask)
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

torch.save(detector_trt.state_dict(), 'final_trt.pth')
