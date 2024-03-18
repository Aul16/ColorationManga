from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

import pandas as pd


#On crée un dataset custom à partir des images du dossier 'dataset'
class LoadDataset(Dataset):
    def __init__(self, img_csv, img_size, PATH_BW, PATH_RGB):
        self.PATH_BW = PATH_BW
        self.PATH_RGB = PATH_RGB
        self.img_csv = pd.read_csv(img_csv)
        self.transform = transforms.Resize(img_size)

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, index):
        img_name = self.img_csv.iloc[index, 0]
        image = [read_image(f"{self.PATH_BW}/{img_name}"), read_image(f"{self.PATH_RGB}/{img_name}", ImageReadMode.RGB)]
        # On reshape les données :

        image = [self.transform(img) for img in image]

        #On normalise les données
        image = [(img - 127.5)/127.5 for img in image]
        return image[0], image[1]