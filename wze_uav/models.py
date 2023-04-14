# Author: Simon Ecke
# Date: 03.04.2023
# Code Repository: https://github.com/simon-ecke/wze-uav_dl
#
# Description: This code defines several functions to create different pre-trained deep learning models, including EfficientNet (B0, B2, B7, V2_L) and ResNet152. These functions accept parameters such as the output shape, whether to freeze or unfreeze the base model layers, the dropout rate, and the device to use for computation. They return the created model.


import torch
import torchvision
from torch import nn


def create_effnetb0(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # Get base model and pre-trained weights as well as auto transforms
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    # Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # Change the model's classifier head and add dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1280, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    model.name = "effnet_b0"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnetb2(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # Get base model and pre-trained weights as well as auto transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    
    # Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # Change the model's classifier head and add dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1408, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    model.name = "effnet_b2"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnetb7(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # Get base model and pre-trained weights as well as auto transforms
    weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = torchvision.models.efficientnet_b7(weights=weights).to(device)
    
    # Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # Change the model's classifier head and add dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=2560, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    model.name = "effnet_b7"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnet_v2_l(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # Get base model and pre-trained weights as well as auto transforms
    weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = torchvision.models.efficientnet_v2_l(weights=weights).to(device)
    
    # Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # Change the model's classifier head and add dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    model.name = "effnet_v2_l"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_resnet152(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # Get base model and pre-trained weights as well as auto transforms
    weights = torchvision.models.ResNet152_Weights.DEFAULT
    auto_transforms = weights.transforms()
    model = torchvision.models.resnet152(weights=weights).to(device)
    
    # Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # Change the model's classifier head and add dropout 
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate),
                             nn.Linear(in_features=model.fc.in_features,
                                       out_features=output_shape)).to(device)

    
    model.name = "resnet152"
    
    print(f"[INFO] Created new {model.name} model.")
    return model