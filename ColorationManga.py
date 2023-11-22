import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from layers.decoder import Decoder
from layers.encoder import Encoder

wandb.login()

run = wandb.init(
    # Nom du Projet
    project="Coloration Manga",
    # Sauvegarde des hyperparamètres
    config={
    "learning_rate": 0.001,
    "epochs": 50,
    })

BATCH_SIZE = 8
IMG_SHAPE = (1024, 768)

PATH_RGB = "./dataset/rgb"
PATH_BW = "./dataset/bw"
CSV_PATH = "./images_Train_DGX.csv"
SAVE_PATH = "./saves"


#On crée un dataset custom à partir des images du dossier 'dataset'
class TrainDataset(Dataset):
    def __init__(self, img_csv, img_size):
        self.img_csv = pd.read_csv(img_csv)
        self.transform = transforms.Resize(img_size)

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, index):
        img_name = self.img_csv.iloc[index, 0]
        image = [read_image(f"{PATH_BW}/{img_name}"), read_image(f"{PATH_RGB}/{img_name}", ImageReadMode.RGB)]
        # On reshape les données :

        image = [self.transform(img) for img in image]

        #On normalise les données
        image = [(img - 127.5)/127.5 for img in image]
        return image[0], image[1]


data = TrainDataset(CSV_PATH, IMG_SHAPE)
train_size = int(0.9 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

#On charge notre dataset dans un dataloader
x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


class AutoEncoderBW(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.encoder = Encoder(n)
        self.decoder = Decoder(n)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

model = AutoEncoderBW(8).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 50
for i in range(epoch):

    epoch_loss_train = 0
    for j, batch in tqdm(enumerate(x_train)):
        optimizer.zero_grad()
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()

    epoch_loss_val = 0
    for j, batch in tqdm(enumerate(x_test)):
        x, y = batch
        x_hat = model(x.to(device))
        loss = loss_function(x_hat, x.to(device))*10
        epoch_loss_val += loss.item()

    wandb.log({"Train Loss": epoch_loss_train, "Test Loss": epoch_loss_val, "Validation Image": wandb.Image(torch.cat((x[0], x_hat[0]), dim=2).detach().numpy())})

    torch.save(model, f"{SAVE_PATH}/model/model{i}.pth")
    torch.save(model.encoder, f"{SAVE_PATH}/encoder/encoder{i}.pth")
