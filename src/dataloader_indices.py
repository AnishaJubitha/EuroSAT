from torch.utils.data import Dataset, DataLoader
from src.dataloader import *
import numpy as np
import rasterio
import torch

class SpectralIndicesDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def safe_divide(self, numerator, denominator):
        return numerator/(denominator + 1e-8)
    
    def compute_rgb_indices(self, ms_image):
        blue = ms_image[1]
        green = ms_image[2]
        red = ms_image[3]
        nir = ms_image[7]
        swir1 = ms_image[10]

        ndvi = self.safe_divide(nir-red, nir+red)
        ndwi = self.safe_divide(green-nir, green+nir)
        ndbi = self.safe_divide(swir1-nir, swir1+nir)
        ndmi = self.safe_divide(nir-swir1, nir+swir1)

        bsi = self.safe_divide((swir1+red)-(nir+blue),
                               (swir1+red)+(nir+blue))
        
        image = np.stack(
            [red, green, blue, ndvi, ndwi, ndbi, ndmi, bsi],
            axis=0
        )

        return image
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        path = row['path_ms']
        label = row['idx']

        with rasterio.open(path) as src:
            ms_image = src.read().astype(np.float32)

        ms_image = ms_image/10000.0     # to scale images between 0 to 1

        image = self.compute_rgb_indices(ms_image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_dataloader_indices():
    
    train_df, val_df, test_df = load_dataset_splits()

    train_ind_dataset = SpectralIndicesDataset(train_df)

    val_ind_dataset = SpectralIndicesDataset(val_df)

    test_ind_dataset = SpectralIndicesDataset(test_df)

    train_ind_loader = DataLoader(train_ind_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_ind_loader = DataLoader(val_ind_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    test_ind_loader = DataLoader(test_ind_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_ind_loader, val_ind_loader, test_ind_loader