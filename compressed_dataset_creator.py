import torch
import os

from layers.autoencoderbw import AutoEncoderBW
from layers.autoencoderrgb import AutoEncoderRGB
from datasets_loader.image_dataloader import AutoEncoderDataset

IMG_SHAPE = (1024, 768)

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files

PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images.csv"
SAVE_PATH = "./saves"
NEW_SAVE_PATH = "./compressed_dataset"

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

##############################################################################################################
#
#  Create second dataset for UResNet (compressed images)
#
##############################################################################################################

encodebw = torch.load(f"{SAVE_PATH}/BW/encoder4.pth")
encodergb = torch.load(f"{SAVE_PATH}/RGB/encoder4.pth")

data = AutoEncoderDataset(CSV_PATH, IMG_SHAPE, PATH_BW, PATH_RGB)

for i, imgs in enumerate(data):
    bw, rgb = imgs
    bw = bw.unsqueeze(0)
    rgb = rgb.unsqueeze(0)
    compressed_bw = encodebw(bw.to(device))[0]
    compressed_rgb = encodergb(rgb.to(device))[0]
    torch.save(compressed_bw, f"{NEW_SAVE_PATH}/bw/tensor{i}.pt")
    torch.save(compressed_rgb, f"{NEW_SAVE_PATH}/rgb/tensor{i}.pt")
    if i > 50:
        break

# Create CSV file
import os
tensors = os.listdir(f"{NEW_SAVE_PATH}/bw/")

with open("./compressed_data.csv", "w", newline='') as csvfile:
    for tensor_name in tensors:
        csvfile.writelines(f"{tensor_name}\n")

