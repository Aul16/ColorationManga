import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image

import os

from layers.uresnet import UResNet

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

IMG_SHAPE = (512, 512)
PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"

model = torch.load(f"weights.pth", map_location=device)

image = read_image("./image.jpg")
image = transforms.Resize(IMG_SHAPE)(image)
image = (image - 127.5) / 127.5
image = image.unsqueeze(0).to(device)

fake_RGB = model(image)

save_image(fake_RGB, "./image_colorized.png")
