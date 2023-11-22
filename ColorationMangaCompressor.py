import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb

from layers.autoencoderbw import AutoEncoderBW
from layers.autoencoderrgb import AutoEncoderRGB
from datasets_loader.image_dataloader import AutoEncoderDataset

wandb.login()

BATCH_SIZE = 8
IMG_SHAPE = (1024, 768)

PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images.csv"
SAVE_PATH = "./saves"

ENCODER_CHANNEL_OUTPUT = 8
DECODER_CHANNEL_INTPUT = 8

data = AutoEncoderDataset(CSV_PATH, IMG_SHAPE, PATH_BW, PATH_RGB)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))





##############################################################################################################
#
#  Train Encoder for BW images
#
##############################################################################################################

run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="Encoder",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": 50,
    })

model = AutoEncoderBW(ENCODER_CHANNEL_OUTPUT).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 5
for i in range(epoch):
    epoch_loss_train = 0
    j=0
    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
        j+=1
        if j > 3:
            break

    epoch_loss_val = 0
    j=0
    for batch in x_test:
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        epoch_loss_val += loss.item()
        j+=1
        if j > 3:
            break


    wandb.log({"Train Loss": epoch_loss_train, "Test Loss": epoch_loss_val, "Validation Image": wandb.Image(torch.cat((x[0], x_hat[0].cpu()), dim=2).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/BW/model{i}.pth")
    torch.save(model.encoder, f"{SAVE_PATH}/BW/encoder{i}.pth")

wandb.finish()

##############################################################################################################





##############################################################################################################
#
#  Train Decoder for RGB images
#
##############################################################################################################

run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="Decoder",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": 50,
    })

model = AutoEncoderRGB(DECODER_CHANNEL_INTPUT).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 5
for i in range(epoch):

    epoch_loss_train = 0
    j = 0
    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
        j+=1
        if j > 3:
            break

    epoch_loss_val = 0
    j = 0
    for batch in x_test:
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        epoch_loss_val += loss.item()
        j+=1
        if j > 3:
            break

    wandb.log({"Train Loss": epoch_loss_train, "Test Loss": epoch_loss_val, "Validation Image": wandb.Image(torch.cat((y[0], y_hat[0].cpu()), dim=2).permute(1,2,0).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/RGB/model{i}.pth")
    torch.save(model.encoder, f"{SAVE_PATH}/RGB/encoder{i}.pth")
    torch.save(model.decoder, f"{SAVE_PATH}/RGB/decoder{i}.pth")

wandb.finish()

##############################################################################################################
