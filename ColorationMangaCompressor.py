import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.autoencoderbw import AutoEncoderBW
from layers.autoencoderrgb import AutoEncoderRGB
from datasets_loader.image_dataloader import AutoEncoderDataset

wandb.login()

PATH = os.path.dirname(os.path.abspath(__file__))  # Get the path of the files
os.chdir(PATH)  # Change the current working directory to the path of the files
torch.cuda.empty_cache()

os.makedirs("./saves", exist_ok=True)
os.makedirs("./saves/BW", exist_ok=True)
os.makedirs("./saves/RGB", exist_ok=True)
os.makedirs("./saves/uresnet", exist_ok=True)
os.makedirs("./saves/uresnet_no_comp", exist_ok=True)
os.makedirs("./compressed_dataset", exist_ok=True)
os.makedirs("./compressed_dataset/bw", exist_ok=True)
os.makedirs("./compressed_dataset/rgb", exist_ok=True)

BATCH_SIZE = 32
IMG_SHAPE = (1024, 768)

PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images.csv"
SAVE_PATH = "./saves"

ENCODER_CHANNEL_OUTPUT = 16
DECODER_CHANNEL_INPUT = 48

data = AutoEncoderDataset(CSV_PATH, IMG_SHAPE, PATH_BW, PATH_RGB)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)




##############################################################################################################
#
#  Train Encoder for BW images
#
##############################################################################################################

epoch = 3
run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="Encoder",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": epoch,
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE",
    "architecture": "Encoder",
    "optimizer": "Adam",
    "encoder_channel_output": ENCODER_CHANNEL_OUTPUT
    })

model = AutoEncoderBW(ENCODER_CHANNEL_OUTPUT).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epoch):

    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        loss.backward()
        optimizer.step()
        wandb.log({"Train Loss": loss.item(), "Epoch": i})

    torch.cuda.empty_cache()
    
    for batch in x_test:
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        wandb.log({"Test Loss": loss.item(), "Epoch": i})
        
    torch.cuda.empty_cache()

    wandb.log({"Validation Image": wandb.Image(torch.cat((x[0], x_hat[0].cpu()), dim=2).detach().numpy()), "Epoch": i})

    torch.save(model, f"{SAVE_PATH}/BW/model{i}.pth")
    torch.save(model.encoder, f"{SAVE_PATH}/BW/encoder{i}.pth")
    torch.save(model.decoder, f"{SAVE_PATH}/BW/decoder{i}.pth")

wandb.finish()
torch.cuda.empty_cache()


##############################################################################################################


# BATCH_SIZE  for rgb images
BATCH_SIZE = 12
#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


##############################################################################################################
#
#  Train Decoder for RGB images
#
##############################################################################################################

epoch = 3
run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    name="Decoder",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": epoch,
    "batch_size": BATCH_SIZE,
    "loss_function": "MSE",
    "architecture": "Decoder",
    "optimizer": "Adam",
    "decoder_channel_input": DECODER_CHANNEL_INPUT
    })

model = AutoEncoderRGB(DECODER_CHANNEL_INPUT).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epoch):

    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        loss.backward()
        optimizer.step()
        wandb.log({"Train Loss": loss.item(), "Epoch": i})
        
    torch.cuda.empty_cache()

    for batch in x_test:
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        wandb.log({"Test Loss": loss.item(), "Epoch": i})
        
    torch.cuda.empty_cache()

    wandb.log({"Epoch": i,"Validation Image": wandb.Image(torch.cat((y[0], y_hat[0].cpu()), dim=2).permute(1,2,0).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/RGB/model{i}.pth")
    torch.save(model.encoder, f"{SAVE_PATH}/RGB/encoder{i}.pth")
    torch.save(model.decoder, f"{SAVE_PATH}/RGB/decoder{i}.pth")

wandb.finish()
torch.cuda.empty_cache()
##############################################################################################################
