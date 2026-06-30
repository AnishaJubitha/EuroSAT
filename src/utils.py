from src.config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch


def map_labels(CLASSES):

    class_to_idx = {
        class_name : index 
        for index, class_name in enumerate(CLASSES)
    }

    idx_to_class = {
        index : class_name 
        for index, class_name in enumerate(CLASSES)
    }

    return class_to_idx, idx_to_class


def count_files(CLASSES, path, extn):

    file_counts = []

    for c in CLASSES:
        folder = path / c
        imgs = list(folder.glob(f"*.{extn}"))
        #print(imgs)

        file_counts.append({ c : len(imgs)})

    return file_counts



# Build DataFrame
def build_df(CLASSES, path, extn):
    li = []

    for cls in CLASSES:
        class_file = path / cls

        files = sorted(class_file.glob(f'*.{extn}'))

        for file_path  in files:
            li.append({
                'path' : str(file_path),
                'label' : cls,
                'image_id' : file_path.stem
            })

    return pd.DataFrame(li)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print('GPU : ', torch.cuda.get_device_name(0))
        print('CUDA version : ', torch.version.cuda)

    return device


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        
def save_history(history):
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history_df) + 1)

    history_df = history_df[
        ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']
    ]

    history_df.to_csv(
        METRICS_PATH / 'rgb_model_history.csv',
        index=False
    )

