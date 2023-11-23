import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb
import os

from layers.uresnet import UResNet
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
    "epochs": 50,
    })

model = UResNet(n=16, channel_in=ENCODER_CHANNEL_OUTPUT, channel_out=DECODER_CHANNEL_INTPUT).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 100
for i in range(epoch):

    epoch_loss_train = 0
    for batch in x_train:
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()


    epoch_loss_val = 0
    for batch in x_test:
        x, y = batch
        y_hat = model(y.to(device))
        loss = loss_function(y_hat, y.to(device))*10
        epoch_loss_val += loss.item()


    wandb.log({"Train Loss": epoch_loss_train, "Test Loss": epoch_loss_val, "Validation Image": wandb.Image(torch.flatten(torch.cat((y[0], y_hat[0]), dim=2), end_dim=1).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/uresnet/uresnet{i}.pth")

wandb.finish()