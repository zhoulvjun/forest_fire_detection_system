import cv2
import sys
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, draw_mask
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UNET  # seg 4 classes
import matplotlib.pyplot as plt

from torch2trt import torch2trt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_valid = transforms.Compose([
    transforms.Resize((255, 255), interpolation=2),
    transforms.ToTensor()
]
)

img = Image.open("../datas/database/000023.jpg")
img_cv = cv2.imread("../datas/database/000023.jpg")
img_ = transform_valid(img).unsqueeze(0).to(DEVICE)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

# original model
pre = model(img_)
cv_mask = tensor_to_cv(pre[0].cpu())
cv2.imshow("cv", cv_mask)
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

# optimied model with thesorrt
init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x])

pre = detector_trt(img_)
cv_mask = tensor_to_cv(pre[0].cpu())
cv2.imshow("cv", cv_mask)
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

torch.save(detector_trt.state_dict(), 'final_trt.pth')
