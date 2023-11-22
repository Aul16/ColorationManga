from torch.utils.data import Dataset
import torch

import pandas as pd


#On crée un dataset custom à partir des images du dossier 'dataset'
class UResNetDataset(Dataset):
    def __init__(self, img_csv, PATH_BW, PATH_RGB):
        self.PATH_BW = PATH_BW
        self.PATH_RGB = PATH_RGB
        self.tensor_csv = pd.read_csv(img_csv)

    def __len__(self):
        return len(self.tensor_csv)

    def __getitem__(self, index):
        tensor_name = self.tensor_csv.iloc[index, 0]
        tensor = [torch.load(f"{self.PATH_BW}/{tensor_name}"), torch.load(f"{self.PATH_RGB}/{tensor_name}")]
        
        return tensor[0], tensor[1]