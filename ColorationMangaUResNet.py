import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.uresnet import UResNet
from layers.autoencoderbw import AutoEncoderBW
from layers.autoencoderrgb import AutoEncoderRGB
from datasets_loader.compressed_dataloader import UResNetDataset

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files

PATH_RGB = "./compressed_dataset/rgb"
PATH_BW = "./compressed_dataset/bw"
CSV_PATH = "./compressed_data.csv"
SAVE_PATH = "./saves"

wandb.login()

BATCH_SIZE = 8

ENCODER_CHANNEL_OUTPUT = 8
DECODER_CHANNEL_INTPUT = 8

data = UResNetDataset(CSV_PATH, PATH_BW, PATH_RGB)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))





##############################################################################################################
#
#  Train UResNet for RGB images
#
##############################################################################################################

run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="UResNet",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": 100,
    })

model = UResNet(n=16, channel_in=ENCODER_CHANNEL_OUTPUT, channel_out=DECODER_CHANNEL_INTPUT).to(device)
decoderbw = torch.load(f"{SAVE_PATH}/BW/decoder4.pth").to(device)
decoderrgb = torch.load(f"{SAVE_PATH}/RGB/decoder4.pth").to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 100
for i in range(epoch):

    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        loss.backward()
        optimizer.step()
        wandb.log({"Train Loss": loss.item(), "Epoch": i})

    for batch in x_test:
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        wandb.log({"Test Loss": loss.item(), "Epoch": i})


    image_bw = decoderbw(y.to(device))
    image_bw = image_bw.expand(-1, 3, -1, -1)
    image_rgb = decoderrgb(y_hat)
    wandb.log({"Epoch": i, "Validation Image": wandb.Image(torch.cat((image_bw[0], image_rgb[0]), dim=2).permute(1,2,0).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/uresnet/uresnet{i}.pth")

wandb.finish()