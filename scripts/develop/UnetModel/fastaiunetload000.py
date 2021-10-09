import cv2
import sys
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, show_cv_image, draw_mask
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastaiunetmodel000 import pureunet  # seg 4 classes
import matplotlib.pyplot as plt

from torch2trt import torch2trt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_valid = transforms.Compose([
    transforms.Resize((255, 255), interpolation=2),
    transforms.ToTensor()
]
)

img = Image.open("../datas/Smoke_segmentation/testing/image_00001.jpg")
img_cv = cv2.imread("../datas/Smoke_segmentation/testing/image_00001.jpg")
img_ = transform_valid(img).unsqueeze(0).to(DEVICE)

model = pureunet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

# original model
pre = model(img_)
cv_mask = tensor_to_cv(pre[0].cpu())
show_cv_image(cv_mask,"cv")

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
show_cv_image(masked_img,"cv")

# optimied model with thesorrt
init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x])

pre = detector_trt(img_)
cv_mask = tensor_to_cv(pre[0].cpu())
show_cv_image(cv_mask,"cv")

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)
show_cv_image(masked_img,"cv")

torch.save(detector_trt.state_dict(), 'final_trt.pth')
