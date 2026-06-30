import matplotlib.pyplot as plt
from src.config import *
from src.dataloader import *
from src.utils import *
import pandas as pd
import numpy as np


# Plot RGB and MS Data
def plot_bar_graph(key, val, title, path=None):

    plt.bar(key, val)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=12)

    plt.savefig(
       path , dpi=300
    )

    plt.show()

    return plt
    

def ms_visualisation_uint8(ms_bands):

    ms = np.stack(ms_bands, axis=-1)
    ms = ms.astype('float32')
    ms8 = (ms - ms.min()) / (ms.max() - ms.min() + 1e-8)

    return ms8

def plot_training_curves(history, x, y, prefix):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history[x], label=x.upper())
    plt.plot(epochs, history[y], label=y.upper())
    plt.xlabel('EPOCHS')

    if 'loss' in y:
        fig_name = prefix + '_model_loss.png'
        plt.ylabel('LOSS')
        plt.title( prefix.upper() + ' Model - Training & Validation Loss')
    elif 'acc' in y:
        fig_name = prefix + '_model_acc.png'
        plt.ylabel('ACCURACY')
        plt.title( prefix.upper() + ' Model - Training & Validation Accuracy')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(METRICS_PATH / fig_name, dpi=300)
    plt.show()

def plot_confusion_matrix(cm, title, CLASSES, path_file):
    plt.figure(figsize=(10, 8))

    plt.imshow(cm)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.xticks(
        ticks=np.arange(NUM_CLASSES),
        labels=CLASSES,
        rotation=45,
        ha='right'
    )

    plt.yticks(
        ticks=np.arange(NUM_CLASSES),
        labels=CLASSES
    )

    plt.colorbar()

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.savefig(
        path_file, dpi=300
    )

    plt.show()





        


