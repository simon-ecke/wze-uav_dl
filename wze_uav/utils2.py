
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler




def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str,
               file_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A folder inside the target_dir named after the model name.
    file_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="efficientnet_b0",
               file_name="01_model_32batch.pth")
    """
    # Create target directory
    target_dir_path = Path(f"{target_dir}/{model_name}")
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)
    
    # Create model save path
    assert file_name.endswith(".pth") or file_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / file_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
             f=model_save_path)
     
    
# from https://github.com/clcarwin/focal_loss_pytorch 


class FocalLoss(nn.Module):
    def __init__(self, weights=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha
        if isinstance(alpha, (float, int,)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.weights * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# modified FocalLoss to allow one-hot encoded labels as input
class FocalLoss2(nn.Module):
    def __init__(self, weights=0, alpha=None, size_average=True):
        super(FocalLoss6, self).__init__()
        self.weights = weights
        self.alpha = alpha
        if isinstance(alpha, (float, int,)):
            self.alpha = torch.Tensor([alpha] * num_classes)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, target.size(-1))

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.argmax(dim=1, keepdim=True))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.argmax(dim=1, keepdim=True).data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.weights * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CustomExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, min_lr, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super(CustomExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * math.pow(self.gamma, self.last_epoch), self.min_lr)
                for base_lr in self.base_lrs]
    
    
class CustomStepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=0):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(CustomStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.step_size:
            return [base_lr for base_lr in self.base_lrs]
        else:
            lr = [base_lr * self.gamma ** (self.last_epoch // self.step_size) for base_lr in self.base_lrs]
            return [max(base_lr, self.min_lr) for base_lr in lr]