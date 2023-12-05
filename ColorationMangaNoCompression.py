from re import S
from numpy import real, save
import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os
import kornia

from layers.uresnet import UResNet
from datasets_loader.image_lab_dataloader import AutoEncoderDataset
from layers.disciminator import Discriminator

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.cuda.empty_cache()

IMG_SHAPE = (512, 512)
PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images.csv"
SAVE_PATH = "./saves"
EPOCH = 10

wandb.login()

BATCH_SIZE = 1
BATCH_TEST = 16

ENCODER_CHANNEL_OUTPUT = 1
DECODER_CHANNEL_INPUT = 2

data = AutoEncoderDataset(CSV_PATH, IMG_SHAPE, PATH_BW, PATH_RGB)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_TEST, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))


##############################################################################################################
#
#  Train UResNet for RGB images
#
##############################################################################################################

run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="UResNet LAB",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE + BCE",
    "architecture": "UResNet",
    "dataset": "dataset",
    "optimizer": "Adam",
    "encoder_channel_output": ENCODER_CHANNEL_OUTPUT,
    "decoder_channel_input": DECODER_CHANNEL_INPUT
    })

model = UResNet(n=64, channel_in=ENCODER_CHANNEL_OUTPUT, channel_out=DECODER_CHANNEL_INPUT).to(device)
discriminator = Discriminator(3, 64).to(device)  # Channel in: 3 (LAB)

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()
optimizer_unet = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.0005)

for i in range(EPOCH):
    
    iter_test = iter(x_test)

    model.train()
    for j, batch in enumerate(x_train):
        ############################################################
        real_rgb, real_L, real_AB = batch
        
        real_L = real_L.to(device)
        real_AB = real_AB.to(device)
        
        fake_AB = model(real_L)

        # Entraine le discriminateur
        discriminator.zero_grad()
        
        real_pred = discriminator(real_AB, real_L)
        fake_pred = discriminator(fake_AB.detach(), real_L)

        fake_labels = torch.zeros_like(fake_pred)
        fake_labels = fake_labels.type_as(real_AB).to(device)
        real_labels = torch.ones_like(real_pred)
        real_labels = real_labels.type_as(real_AB).to(device)

        ### On calcule la loss pour les vraies et les fausses images

        fake_loss = loss_bce(fake_pred, fake_labels)
        real_loss = loss_bce(real_pred, real_labels)

        disc_loss = (fake_loss + real_loss)/2

        ### Backpropagation

        disc_loss.backward()
        optimizer_disc.step()

        ############################################################

        ### Entraine le UResNet
        
        model.zero_grad()

        fake_pred = discriminator(fake_AB, real_L)

        real_labels = torch.ones_like(fake_pred)
        real_labels = real_labels.type_as(real_AB).to(device)

        ### On calcule la loss pour le générateur

        unet_mse = loss_mse(fake_AB, real_AB)*10
        unet_disc_loss = loss_bce(fake_pred, real_labels)

        unet_loss = unet_mse + unet_disc_loss
        
        ### Backpropagation

        unet_loss.backward()
        optimizer_unet.step()

        wandb.log({"UResNet MSE Loss": unet_mse.item(),
                "Epoch": i,
                "Discriminator Loss": disc_loss.item(),
                "UResNet Discriminator Loss": unet_disc_loss.item()},
                  commit=False)
        
        if j % 1000 == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter_test)
                real_RGB, real_L, real_AB = batch
                fake_AB = model(real_L.to(device))
                loss_val = loss_mse(fake_AB, real_AB.to(device))*10
                wandb.log({"Test Loss": loss_val.item(), "Epoch": i},
                          commit=False)
    
                real_L = (real_L+1)*50
                fake_AB = fake_AB * 128
                fake_RGB = kornia.color.lab_to_rgb(torch.cat((real_L, fake_AB.cpu()), dim=1))*255
                real_L = real_L.expand(-1, 3, -1, -1)*2.55
                image1 = torch.cat((real_L[0].cpu(), real_RGB[0].cpu(), fake_RGB[0].cpu()), dim=2).detach()
                image2 = torch.cat((real_L[1].cpu(), real_RGB[1].cpu(), fake_RGB[1].cpu()), dim=2).detach()
                image3 = torch.cat((real_L[2].cpu(), real_RGB[2].cpu(), fake_RGB[2].cpu()), dim=2).detach()
                wandb.log({"Epoch": i, "Validation Image": wandb.Image(torch.cat((image1, image2, image3), dim=1).permute(1,2,0).numpy())},
                          commit=False)
            model.train()
    
        wandb.log({"Epoch": i}, commit=True)
        
    torch.save(model, f"{SAVE_PATH}/uresnet_no_comp/uresnet{i}.pth")
    torch.save(discriminator, f"{SAVE_PATH}/uresnet_no_comp/discriminator{i}.pth")
    os.system(f"rm -rf {SAVE_PATH}/uresnet_no_comp/uresnet{i-3}.pth")  # Keep 3 last models
    os.system(f"rm -rf {SAVE_PATH}/uresnet_no_comp/discriminator{i-3}.pth")  # Keep 3 last models

wandb.finish()
torch.cuda.empty_cache()