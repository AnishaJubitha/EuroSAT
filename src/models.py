import torch.nn as nn
from torchvision import models
from src.config import *
from src.utils import *
import torch

def resnet18_rgb_model(pretrained=True):

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model


def resnet18_ms_model(pretrained=True):

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=NUM_BANDS,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    if pretrained:
        with torch.no_grad():
            old_weights = old_conv.weight

            mean_weights = old_weights.mean(dim=1, keepdim=True)

            model.conv1.weight = nn.Parameter(
                mean_weights.repeat(1, NUM_BANDS, 1, 1)
            )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model

def load_model(path, model_name, pretrained=False):

    DEVICE = get_device()

    if model_name == 'RGB':
        model = resnet18_rgb_model(pretrained=pretrained)
    elif model_name == 'MS':
        model = resnet18_ms_model(pretrained=pretrained)
    elif model_name == 'IND':
        model = resnet18_ind_model(pretrained=pretrained)
    else:
        raise ValueError("Model name must be 'RGB', 'MS', 'IND'.")

    model.load_state_dict(
        torch.load(path, map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()

    return model

def resnet18_ind_model(in_channels=8, pretrained=True):

    '''
    Input Channels = [R G B NDVI NDWI NDBI NDMI BSI]

    '''

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    else:
        model = models.resnet18(weights=None)

    old_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False
    )

    if pretrained:
        with torch.no_grad():
            model.conv1.weight[:, :3, :, :] = old_conv1.weight  # Copying pretrained weights from the first 3 RGB Channels

            mean_rgb_weight = old_conv1.weight.mean(dim=1, keepdim=True)

            for ch in range(3, in_channels):
                model.conv1.weight[:, ch:ch+1, :, :] = mean_rgb_weight     # Using the RGB mean for the exta channels

    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=NUM_CLASSES)
    
    return model
