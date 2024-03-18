import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.uresnet import UResNet
from datasets_loader.image_dataloader import LoadDataset
from layers.disciminator import Discriminator

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.manual_seed(42)
torch.cuda.empty_cache()

IMG_SHAPE = (512, 512)
PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./dataset/images.csv"
EPOCH = 7

wandb.login()

BATCH_SIZE = 16
BATCH_TEST = 16

MODEL_INPUT_CHANNEL = 1
MODEL_OUTPUT_CHANNEL = 3
LEARNING_RATE = 0.001

data = LoadDataset(CSV_PATH, IMG_SHAPE, PATH_BW, PATH_RGB)
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
    name="UResNet",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE + BCE",
    "architecture": "UResNet",
    "dataset": "dataset",
    "optimizer": "Adam"
    })

model = UResNet(n=64, channel_in=MODEL_INPUT_CHANNEL, channel_out=MODEL_OUTPUT_CHANNEL).to(device)
discriminator = Discriminator(4, 64).to(device)

#model = torch.load(f"weights.pth", map_location=device)                  # Load the model if we want to continue the training
#discriminator = torch.load(f"discriminator.pth", map_location=device)    # Load the discriminator if we want to continue the training

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()
optimizer_unet = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE/2)

for i in range(EPOCH):
    
    iter_test = iter(x_test)

    model.train()
    for j, batch in enumerate(x_train):
        ############################################################
        real_bw, real_RGB = batch
        
        real_bw = real_bw.to(device)
        real_RGB = real_RGB.to(device)
        
        fake_RGB = model(real_bw)

        # Entraine le discriminateur
        discriminator.zero_grad()
        
        real_pred = discriminator(real_RGB, real_bw)
        fake_pred = discriminator(fake_RGB.detach(), real_bw)

        fake_labels = torch.zeros_like(fake_pred)
        fake_labels = fake_labels.type_as(real_RGB).to(device)
        real_labels = torch.ones_like(real_pred)
        real_labels = real_labels.type_as(real_RGB).to(device)

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

        fake_pred = discriminator(fake_RGB, real_bw)

        real_labels = torch.ones_like(fake_pred)
        real_labels = real_labels.type_as(real_RGB).to(device)

        ### On calcule la loss pour le générateur

        unet_mse = loss_mse(fake_RGB, real_RGB)*10
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
        
        if j % 100 == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter_test)
                real_bw, real_RGB = batch
                fake_RGB = model(real_bw.to(device))
                loss_val = loss_mse(fake_RGB, real_RGB.to(device))*10
                wandb.log({"Test Loss": loss_val.item(), "Epoch": i},
                          commit=False)
    
                real_bw = real_bw.expand(-1, 3, -1, -1)
                image1 = torch.cat((real_bw[0].cpu(), real_RGB[0].cpu(), fake_RGB[0].cpu()), dim=2).detach()
                image2 = torch.cat((real_bw[1].cpu(), real_RGB[1].cpu(), fake_RGB[1].cpu()), dim=2).detach()
                image3 = torch.cat((real_bw[2].cpu(), real_RGB[2].cpu(), fake_RGB[2].cpu()), dim=2).detach()
                wandb.log({"Epoch": i, "Validation Image": wandb.Image(torch.cat((image1, image2, image3), dim=1).permute(1,2,0).numpy())},
                          commit=False)
            model.train()
    
        wandb.log({"Epoch": i}, commit=True)
        
    torch.save(model, "weights.pth")
    torch.save(discriminator, "discriminator.pth")

wandb.finish()
torch.cuda.empty_cache()
