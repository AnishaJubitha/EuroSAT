from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.dataloader import *
from PIL import Image
from src.config import *
from src.utils import *
import rasterio


class FusionDataset(Dataset):
    def __init__(self, dataframe, transform_rgb=None):
        self.dataframe = dataframe.reset_index(drop=True)

        self.transform_rgb = transform_rgb

        CLASSES = [i.name for i in RGB_PATH.iterdir()]

        self.class_to_idx, self.idx_to_class = map_labels(CLASSES)

        if self.transform_rgb is None:
            self.transform_rgb = transforms.Compose([
                transforms.Resize((64, 64)), 
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        rgb_path  = row['path_rgb']
        ms_path = row['path_ms']
        label = row['idx']

        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_image = self.transform_rgb(rgb_image)

        with rasterio.open(ms_path) as src:
            ms_image = src.read().astype(np.float32)

        ms_image = ms_image / 10000.0
        ms_image = torch.tensor(ms_image, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)

        return rgb_image, ms_image, label
    

def get_dataloader_fusion():

    train_df, val_df, test_df = load_dataset_splits()    

    train_dataset = FusionDataset(train_df)
    val_dataset = FusionDataset(val_df)
    test_dataset = FusionDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader






