import torch
import pandas as pd
from src.config import *
from src.visualisation import *
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)


def get_predictions(model, dataloader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            all_preds.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    return all_labels, all_preds


def get_classification_metrics(y_true, y_pred, path_file):

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall' : recall,
        'macro_f1' : f1
    }

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(METRICS_PATH / path_file, 
                        index=False)

    return metrics_df


def get_classification_df(y_true, y_pred, CLASSES, path_file):
    li = classification_report(
        y_true, y_pred, target_names=CLASSES,
        output_dict=True, zero_division=0
    )

    df = pd.DataFrame(li).transpose()

    df.to_csv(
        METRICS_PATH / path_file
        )

    return df


def get_confusion_matrix(y_true, y_pred, CLASSES, path_file):

    #ANI
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, 'CONFUSION METRICS', CLASSES, METRICS_PATH / path_file)

    return 


def plot_all_evaluation(history, model, test_loader, prefix):
    save_history(history)

    # Plotting - Loss and Acc Curves

    plot_training_curves(history, 'train_loss', 'val_loss', prefix)

    plot_training_curves(history, 'train_acc', 'val_acc', prefix)

    train_df, val_df, test_df = load_dataset_splits()

    if prefix == 'fusion':
        y_true, y_pred = get_fusion_predictions(model, test_loader, get_device())

    else:
        y_true, y_pred = get_predictions(model, test_loader, get_device())

    test_metrics_df = get_classification_metrics(y_true, y_pred, prefix + '_test_metrics.csv')

    print(test_metrics_df)

    CLASSES = sorted(train_df['label'].unique())

    classification_df = get_classification_df(y_true, y_pred, CLASSES, prefix + '_classification_report.csv')

    print(classification_df)

    cm = get_confusion_matrix(y_true, y_pred, CLASSES, prefix + '_confusion_matrix.png')


def test_model(model, test_loader, prefix):

    train_df, val_df, test_df = load_dataset_splits()

    y_true, y_pred = get_predictions(model, test_loader, get_device())


    test_metrics_df = get_classification_metrics(y_true, y_pred, prefix + '_test_metrics.csv')

    print(test_metrics_df)

    CLASSES = sorted(test_df['label'].unique())

    classification_df = get_classification_df(y_true, y_pred, CLASSES, prefix + '_classification_report.csv')

    print(classification_df)

    cm = get_confusion_matrix(y_true, y_pred, CLASSES, prefix + '_confusion_matrix.png')


def get_fusion_predictions(model, dataloader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for rgb_images, ms_images, labels in dataloader:
            
            rgb_images = rgb_images.to(device)
            ms_images = ms_images.to(device)
            labels = labels.to(device)

            outputs = model(rgb_images, ms_images)
            _, predictions = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true, y_pred

