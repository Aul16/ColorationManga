from re import S
import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.uresnet import UResNet
from datasets_loader.image_dataloader import AutoEncoderDataset
from layers.disciminator import Discriminator

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.cuda.empty_cache()

PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images.csv"
SAVE_PATH = "./saves"

wandb.login()

BATCH_SIZE = 64

ENCODER_CHANNEL_OUTPUT = 1
DECODER_CHANNEL_INPUT = 3

data = AutoEncoderDataset(CSV_PATH, PATH_BW, PATH_RGB)
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
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE",
    "architecture": "UResNet",
    "dataset": "dataset",
    "optimizer": "Adam",
    "encoder_channel_output": ENCODER_CHANNEL_OUTPUT,
    "decoder_channel_input": DECODER_CHANNEL_INPUT
    })

model = UResNet(n=32, channel_in=ENCODER_CHANNEL_OUTPUT, channel_out=DECODER_CHANNEL_INPUT).to(device)
discriminator = Discriminator(4, 16).to(device)  # Channel in: 3 (RGB), 1 (BW)

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()
optimizer_unet = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

epoch = 100
for i in range(epoch):

    model.train()
    for batch in x_train:
        ############################################################
        real_bw, real_rgb = batch
        fake_rgb = model(real_bw.to(device))

        # Entraine le discriminateur
        discriminator.zero_grad()
        
        real_pred = discriminator(real_rgb.to(device), real_bw.to(device))
        fake_pred = discriminator(fake_rgb, real_bw.to(device))

        fake_labels = torch.zeros_like(fake_pred)
        fake_labels = fake_labels.type_as(real_rgb)
        real_labels = torch.ones_like(real_pred)
        real_labels = real_labels.type_as(real_rgb)

        ### On calcule la loss pour les vraies et les fausses images

        fake_loss = loss_bce(fake_pred, fake_labels)
        real_loss = loss_bce(real_pred, real_labels)

        disc_loss = (fake_loss + real_loss)

        ### Backpropagation

        disc_loss.backward()
        optimizer_disc.step()

        ############################################################

        ### Entraine le UResNet

        model.zero_grad()
        fake_rgb = model(real_bw.to(device))
        fake_pred = discriminator(fake_rgb, real_bw.to(device))

        real_labels = torch.ones_like(fake_pred)
        real_labels = real_labels.type_as(real_rgb)

        ### On calcule la loss pour le générateur

        unet_mse = loss_mse(fake_rgb, real_rgb)*100
        unet_disc_loss = loss_bce(fake_pred, real_labels)

        unet_loss = unet_mse + unet_disc_loss
        
        ### Backpropagation

        unet_loss.backward()
        optimizer_unet.step()

        wandb.log({"UResNet MSE Loss": unet_mse.item(),
                "Epoch": i,
                "Discriminator Loss": disc_loss.item(),
                "UResNet Discriminator Loss": unet_disc_loss.item()})
        
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        for batch in x_test:
            real_bw, real_rgb = batch
            fake_rgb = model(real_bw.to(device))
            loss_val = loss_mse(fake_rgb, real_rgb.to(device))*10
            wandb.log({"Test Loss": loss_val.item(), "Epoch": i})
    torch.cuda.empty_cache()
    
    wandb.log({"Epoch": i, "Validation Image": wandb.Image(torch.cat((real_bw[0].cpu(), real_rgb[0].cpu(), fake_rgb[0].cpu()), dim=2).permute(1,2,0).detach().numpy())})
    torch.save(model, f"{SAVE_PATH}/uresnet_no_comp/uresnet{i}.pth")
    os.system(f"rm -rf {SAVE_PATH}/uresnet/uresnet{i-3}.pth")  # Keep 3 last models

wandb.finish()
torch.cuda.empty_cache()