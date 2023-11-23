import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.uresnet import UResNet
from layers.autoencoderbw import AutoEncoderBW
from layers.autoencoderrgb import AutoEncoderRGB
from datasets_loader.compressed_dataloader import UResNetDataset
from layers.disciminator import Discriminator

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files

PATH_RGB = "./compressed_dataset/rgb"
PATH_BW = "./compressed_dataset/bw"
CSV_PATH = "./compressed_data.csv"
SAVE_PATH = "./saves"

wandb.login()

BATCH_SIZE = 4

ENCODER_CHANNEL_OUTPUT = 8
DECODER_CHANNEL_INPUT = 24

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
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE",
    "architecture": "UResNet",
    "dataset": "compressed",
    "optimizer": "Adam",
    "encoder_channel_output": ENCODER_CHANNEL_OUTPUT,
    "decoder_channel_input": DECODER_CHANNEL_INPUT
    })

model = UResNet(n=16, channel_in=ENCODER_CHANNEL_OUTPUT, channel_out=DECODER_CHANNEL_INPUT).to(device)
decoderbw = torch.load(f"{SAVE_PATH}/BW/decoder4.pth").to(device)
decoderrgb = torch.load(f"{SAVE_PATH}/RGB/decoder4.pth").to(device)
discriminator = Discriminator(32, 16).to(device)  # Channel in: 32 (24 for RGB and 8 for BW)

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()
optimizer_unet = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

epoch = 100
for i in range(epoch):

    j = 0
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
        j += 1
        if j > 3:
            break

    
    j = 0
    for batch in x_test:
        real_bw, real_rgb = batch
        fake_rgb = model(real_bw.to(device))
        loss_val = loss_mse(fake_rgb, real_rgb.to(device))*10
        wandb.log({"Test Loss": loss_val.item(), "Epoch": i})
        j += 1
        if j > 3:
            break


    image_bw = decoderbw(real_bw.to(device))
    image_bw = image_bw.expand(-1, 3, -1, -1)
    image_rgb = decoderrgb(fake_rgb)
    wandb.log({"Epoch": i, "Validation Image": wandb.Image(torch.cat((image_bw[0], image_rgb[0]), dim=2).permute(1,2,0).detach().numpy())})

    #torch.save(model, f"{SAVE_PATH}/uresnet/uresnet{i}.pth")

wandb.finish()