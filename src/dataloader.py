from src.config import *
from src.datasets import *
from torch.utils.data import DataLoader
from src.transforms import *
import pandas as pd

def get_dataloader(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader

def load_dataset_splits():
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    test_df = pd.read_csv(TEST_CSV_PATH)

    val_df = pd.read_csv(VAL_CSV_PATH)

    return train_df, val_df, test_df


def load_rgb_data():

    # Loading CSV
    train_df, val_df, test_df = load_dataset_splits()

    train_transform, eval_transform = get_rgb_transforms(True, RGB_MEAN, RGB_STD)

    train_rgb_dataset = RGBDataset(train_df, train_transform)
    #print(len(train_rgb_dataset))
    val_rgb_dataset = RGBDataset(val_df, eval_transform)
    #print(len(val_rgb_dataset))
    test_rgb_dataset = RGBDataset(test_df, eval_transform)
    #print(len(test_rgb_dataset))
    train_loader, val_loader, test_loader = get_dataloader(train_rgb_dataset, val_rgb_dataset, test_rgb_dataset)

    return train_loader, val_loader, test_loader


def load_ms_data():

    # Loading CSV
    train_df, val_df, test_df = load_dataset_splits()

    train_transform, eval_transform = get_ms_transforms(True, MS_MEAN, MS_STD)

    train_ms_dataset = MSDataset(train_df, train_transform)
    #print(len(train_rgb_dataset))
    val_ms_dataset = MSDataset(val_df, eval_transform)
    #print(len(val_rgb_dataset))
    test_ms_dataset = MSDataset(test_df, eval_transform)
    #print(len(test_rgb_dataset))
    train_loader, val_loader, test_loader = get_dataloader(train_ms_dataset, val_ms_dataset, test_ms_dataset)

    return train_loader, val_loader, test_loader