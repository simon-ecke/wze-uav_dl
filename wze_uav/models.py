import torch
import torchvision
from torch import nn



# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def create_effnetb0(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
    
    # define model
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    # 2. Freeze/unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
        
    # 3. Set seeds
    set_seeds()
    
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1280, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    # 5. Give the model a name
    model.name = "effnet_b0"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnetb2(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
    
    # define model
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    
    # 2. Freeze/unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
        
    # 3. Set seeds
    set_seeds()
    
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1408, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    # 5. Give the model a name
    model.name = "effnet_b2"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnetb7(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
    
    # define model
    model = torchvision.models.efficientnet_b7(weights=weights).to(device)
    
    # 2. Freeze/unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
        
    # 3. Set seeds
    set_seeds()
    
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=2560, # 2048 (b5), 2560 (b7)
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    # 5. Give the model a name
    model.name = "effnet_b7"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_effnet_v2_l(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
    
    # define model
    model = torchvision.models.efficientnet_v2_l(weights=weights).to(device)
    
    # 2. Freeze/unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
        
    # 3. Set seeds
    set_seeds()
    
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # 5. Give the model a name
    model.name = "effnet_v2_l"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_resnet152(output_shape: int, unfreeze: bool, dropout_rate: float, device: torch.device):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.ResNet152_Weights.DEFAULT
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
   
    # define model
    model = torchvision.models.resnet152(weights=weights).to(device)
    
    # 2. Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # 3. Set the seeds
    set_seeds()
    
    # 4. Change the classifier head and add dropout
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate),
                             nn.Linear(in_features=model.fc.in_features,
                                       out_features=output_shape)).to(device)

    
    # 5. Give the model a name
    model.name = "resnet152"
    
    print(f"[INFO] Created new {model.name} model.")
    return model


def create_densenet(output_shape: int, unfreeze: bool, device: torch.device):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.DenseNet201_Weights.DEFAULT
    # get the transforms used to create the pretrained weights
    auto_transforms = weights.transforms()
   
    # define model
    model = torchvision.models.densenet201(weights=weights).to(device)
    
    # 2. Freeze/Unfreeze the base model layers
    for param in model.parameters():
        param.requires_grad = unfreeze
    
    # 3. Set the seeds
    set_seeds()
    
    # 4. Change the classifier head
    
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=output_shape).to(device)
    
    # 5. Give the model a name
    model.name = "resnet152"
    
    print(f"[INFO] Created new {model.name} model.")
    return model