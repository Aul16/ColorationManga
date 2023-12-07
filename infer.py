from re import S
from numpy import real, save
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image

import wandb
import os
import kornia

from layers.uresnet import UResNet
from datasets_loader.image_lab_dataloader import AutoEncoderDataset

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.cuda.empty_cache()

IMG_SHAPE = (512, 512)
PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
SAVE_PATH = "./saves"

model = torch.load(SAVE_PATH + "/uresnet_no_comp2/uresnet6.pth", map_location='cpu')

image = read_image("./image.jpg")
image = transforms.Resize(IMG_SHAPE)(image)
print(image.max(), image.min())
lab_img = kornia.color.rgb_to_lab(image/255)
L_img = (lab_img[None, 0, :, :]-50)/50
print(L_img.max(), L_img.min())
L_img = L_img[None, :, :, :]

img_ab = model(L_img)
print(img_ab.max(), img_ab.min())

real_L = (L_img+1)*50
fake_AB = img_ab * 128
fake_RGB = kornia.color.lab_to_rgb(torch.cat((real_L, fake_AB), dim=1))*255
                
save_image(fake_RGB, "./image_colorized.png")
