from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import kornia

import pandas as pd



#On crée un dataset custom à partir des images du dossier 'dataset'
class AutoEncoderDataset(Dataset):
    def __init__(self, img_csv, img_size, PATH_BW, PATH_RGB):
        self.PATH_BW = PATH_BW
        self.PATH_RGB = PATH_RGB
        self.img_csv = pd.read_csv(img_csv)
        self.transform = transforms.Resize(img_size)

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, index):
        img_name = self.img_csv.iloc[index, 0]
        
        RGB = read_image(self.PATH_RGB + "/" + img_name)
        RGB = self.transform(RGB)
        LAB = kornia.color.rgb_to_lab(RGB/255)
        
        return RGB, (LAB[None, 0, :, :]-50)/50, LAB[1:3, :, :]/128