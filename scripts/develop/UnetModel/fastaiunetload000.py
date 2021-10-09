import cv2
import sys
sys.path.append('../../')
from  tools.Tensor_CV2 import tensor_to_cv, show_cv_image, draw_mask
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastaiunetmodel000 import pureunet  # seg 4 classes
import matplotlib.pyplot as plt


def imshow(tensor, title=None):
    image = tensor.clone()  # we clone the tensor to not do changes on it
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()
    if title is not None:
        plt.title(title)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_valid = transforms.Compose([
    transforms.Resize((255, 255), interpolation=2),
    transforms.ToTensor()
]
)

img = Image.open("../datas/Smoke_segmentation/training/image_00001.jpg")
img_cv = cv2.imread("../datas/Smoke_segmentation/training/image_00001.jpg")
img_ = transform_valid(img).unsqueeze(0).to(DEVICE)

model = pureunet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

pre = model(img_)

cv_mask = tensor_to_cv(pre[0].cpu())
print (cv_mask.shape)
show_cv_image(cv_mask,"cv")

masked_img = draw_mask(cv2.resize(img_cv, (255,255)), cv_mask)

show_cv_image(masked_img,"cv")

