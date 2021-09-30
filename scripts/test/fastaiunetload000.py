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

img = Image.open("datas/database/000001.jpg")
img_ = transform_valid(img).unsqueeze(0).to(DEVICE)

model = pureunet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("final.pth"))
model.eval()

pre = model(img_)
imshow(img_[0])
imshow(pre[0])
