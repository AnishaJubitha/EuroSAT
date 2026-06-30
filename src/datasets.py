from torch.utils.data import Dataset
from PIL import Image
import rasterio
import torch


class RGBDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dataframe = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['path_rgb']
        image_label = int(self.dataframe.iloc[idx]['idx'])

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, image_label
    
    

class MSDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dataframe = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['path_ms']
        image_label = int(self.dataframe.iloc[idx]['idx'])

        with rasterio.open(image_path) as src:
            image_ms = src.read()

        image_ms = torch.from_numpy(image_ms).float()

        if self.transform:
            image_ms = self.transform(image_ms)

        return image_ms, image_label