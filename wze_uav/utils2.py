
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, input, target):
        # Compute the cross-entropy loss for each sample in the batch
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return torch.mean(focal_loss)
    

    
class FocalLoss2(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, input, target):
        # Compute the cross-entropy loss for each sample in the batch
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        
        # Compute the softmax probabilities for each class
        p = F.softmax(input, dim=1)
        
        # Compute the probability of the true class for each sample
        pt = p.gather(1, target.view(-1, 1)).squeeze()
        
        # Compute the Focal Loss for each sample
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        # Compute the mean Focal Loss for the batch
        return torch.mean(focal_loss)