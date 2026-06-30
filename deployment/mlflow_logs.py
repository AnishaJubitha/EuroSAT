import torch
import mlflow
import pandas as pd
from pathlib import Path
from src.config import *
from src.models import load_model

def setup_mlflow(experiment_name='EuroSAT Classification'):
    mlflow.set_experiment(experiment_name)


def log_params(config_dict):
    for key, value in config_dict.items():
        mlflow.log_param(key, value)

def log_metrics(metrics_dict, step=None):
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value, step=step)

def log_artifacts(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists():
        mlflow.log_artifacts(str(folder_path))
    else:
        print('Artifacts folder not found.')

def log_checkpoint(model_path):
    model_path = Path(model_path)

    if model_path.exists():
        mlflow.log_artifact(str(model_path), artifact_path='model')
    else:
        print('Model checkpoint not found.')


def log_history(history_path):

    history_path = Path(history_path)

    if not history_path.exists():
        print('Model history not found.')

        return
    
    history_df = pd.read_csv(history_path)

    history_cols = history_df.columns()



def log_metrics_history(model_name, path, method, version):

    mlflow.set_tag('method', method)
    mlflow.set_tag('version', version)

    #model = load_model(CHECKPOINT_PATH, model_name)

    history_df = pd.read_csv(METRICS_PATH / path)

    history_cols = history_df.columns

    metrics_cols = [
        'train_loss',
        'train_acc',
        'val_loss',
        'val_acc',
        'lr'
    ]

    for _, row in history_df.iterrows():

        if 'epoch' in history_df.columns:
            step=int(row['epoch'])
        else:
            step=int(row.name)

        for col in metrics_cols:
            if col in history_df.columns:
                value = row[col]

                if pd.notna(value):
                    mlflow.log_metric(col, float(value), step=step)
    