import torch
import torchvision.transforms as transforms
from PIL import Image
from fastaiunetmodel000 import pureunet # seg 4 classes

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
model = pureunet().to(device=DEVICE)
# checkpoint = torch.load('my_checkpoint.pth.tar')
# model.load_state_dict(torch.load(checkpoint['state_dict']))
model.load_state_dict(torch.load("pureunet.pth")) #, map_location = torch.device("cpu")))
model.eval()
# img = Image.open("datas/fs/001.png")
# # img = img.to(device = DEVICE)
# prediction = model(img)

# # def predict():
    
# #     model = model.to(device = DEVICE)
# #     model.eval()

# #     img = Image.open("datas/fs/001.png")
# #     transform = transforms.Compose([transforms.Resize(255)])
# #     img = img.convert("RGB")
# #     img = transform(img)
# #     img = img.to(device = DEVICE)

# #     with torch.no_grad():
# #         prediction = model(img)

# _, predicted = torch.max(prediction, 1)
# classIndex = predicted[0]

# # if __name__ == "__main__":
#     # predict()





