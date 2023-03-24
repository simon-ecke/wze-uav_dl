from tqdm.auto import tqdm
import torch
import numpy as np

def make_predictions(model: torch.nn.Module, 
                     test_dataloader: torch.utils.data.DataLoader,
                     device: torch.device):
    # 1. Make predictions with trained model
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor


def val_labels_3classes(test_dataset: torch.utils.data.dataset.Dataset):
    torch_tensor = test_dataset.labels
    torch_tensor = torch_tensor.squeeze(dim=1)
    np_arr = torch_tensor.cpu().detach().numpy()
    label_list = []
    for i in np_arr:
        if i <= 25:
            label = 0
            label_list.append(label)
        elif i > 25 and i <= 99:
            label = 1
            label_list.append(label)
        else:
            label = 2
            label_list.append(label)
    array_labels = np.array(label_list)
    val_labels = torch.from_numpy(array_labels)
    return val_labels