# Author: Daniel Bourke
# Code Repository: https://github.com/mrdbourke/pytorch-deep-learning
# 
# Description: The code provides functions for training and validating a PyTorch deep learning model with evaluation metrics for precision, recall, f1 score, and cohen kappa. It uses a multiclass classification setup and allows for early stopping. 
#
# Code modification: Evaluation metrics were added from torchmetrics to calculate precision, recall, f1 score and cohen kappa. Additionally, early stopping and a custom model saver, that saves the trained model if certain conditions apply, were implemented.Tensorboard was replaced  by weights & biases to track evaluation metrics.  

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassCohenKappa
from wze_uav.utils2 import *
import numpy as np
import wandb


# train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               num_classes: int,
               device: torch.device) -> Tuple[float, float]:
    
    """Puts the model in train mode and loops through the train dataloader.

    Args:
        model: Insert defined model. 
        dataloader: Insert train dataloader.
        loss_fn: Insert defined loss function. 
        optimizer: Insert defined optimizer.
        num_classes: Number of classes to be classified (int). 

    Returns:
        train loss and evaluation metrics (train_loss, train_precision, train_recall, train_f1_score, train_acc, train_kappa).
    """
    
    # Put model in train mode
    model.train()
    
    # Setup train loss and other evaluation metrics
    train_loss, train_precision, train_recall, train_f1_score, train_acc, train_kappa = 0, 0, 0, 0, 0, 0
    
    # set up metrics
    pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    kappa_metrics = MulticlassCohenKappa(num_classes=num_classes, weights='quadratic').to(device)
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Loss backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Calculate and accumulate evaluation metrics across all batches
        pre_metrics.update(y_pred, y)
        rec_metrics.update(y_pred, y)
        f1_metrics.update(y_pred, y)
        acc_metrics.update(y_pred, y)
        kappa_metrics.update(y_pred, y)
          
    # Adjust metrics to get average loss and evaluation metrics per epoch 
    train_loss = train_loss / len(dataloader)
    train_precision = pre_metrics.compute().item()
    train_recall = rec_metrics.compute().item()
    train_f1_score = f1_metrics.compute().item()
    train_acc = acc_metrics.compute().item()
    train_kappa = kappa_metrics.compute().item()
    
    # reset metrics for next epoch
    pre_metrics.reset()
    rec_metrics.reset()
    f1_metrics.reset()
    acc_metrics.reset()
    kappa_metrics.reset()
   
    return train_loss, train_precision, train_recall, train_f1_score, train_acc, train_kappa


# validation step
def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_classes: int,
              device: torch.device) -> Tuple[float, float]:
    
    """Puts the model in evaluation mode and loops through the validation dataloader.

    Args:
        model: Insert defined model. 
        dataloader: Insert validation dataloader.
        loss_fn: Insert defined loss function. 
        optimizer: Insert defined optimizer.
        num_classes: Number of classes to be classified (int). 

    Returns:
        validation loss and evaluation metrics (val_loss, val_precision, val_recall, val_f1_score, val_acc, val_kappa).
    """
    
    # Put model in eval mode
    model.eval() 
    
    # Setup validation loss and validation accuracy values
    val_loss, val_precision, val_recall, val_f1_score, val_acc, val_kappa = 0, 0, 0, 0, 0, 0
    
    # set up metrics
    pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    kappa_metrics = MulticlassCohenKappa(num_classes=num_classes, weights='quadratic').to(device)
   
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # Forward pass
            y_pred = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            # Calculate and accumulate evaluation metrics across all batches
            pre_metrics.update(y_pred, y)
            rec_metrics.update(y_pred, y)
            f1_metrics.update(y_pred, y)
            acc_metrics.update(y_pred, y)
            kappa_metrics.update(y_pred, y)
            
    # Adjust metrics to get average loss and evaluation metrics per epoch        
    val_loss = val_loss / len(dataloader)
    val_precision = pre_metrics.compute().item()
    val_recall = rec_metrics.compute().item()
    val_f1_score = f1_metrics.compute().item()
    val_acc = acc_metrics.compute().item()
    val_kappa = kappa_metrics.compute().item()
    

    # reset metrics for next epoch
    pre_metrics.reset()
    rec_metrics.reset()
    f1_metrics.reset()
    acc_metrics.reset()
    kappa_metrics.reset()
   
    return val_loss, val_precision, val_recall, val_f1_score, val_acc, val_kappa


# define train()
def train(model: torch.nn.Module,
          model_name: str,
          n_bands: int,
          batch_size: int,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          num_classes: int,
          epochs: int,
          experiment_num: int,
          device: torch.device,
          writer: None,
          early_stop_patience: int = 5) -> Dict[str, List]:
    
    """Combines the functions train_step() and val_step().

    Args:
        model: Insert defined model. 
        model_name: Define a model name (str).
        batch_size: Defined batch size. 
        train_dataloader: Insert train dataloader.
        val_dataloader: Insert validation dataloader.
        optimizer: Insert defined optimizer.
        lr_scheduler: Insert defined learing rate scheduler.
        loss_fn: Insert defined loss function. 
        num_classes: Number of classes to be classified (int).
        epochs: Number of epochs for the training cycle (int).
        experiment_num: Experiment number (int).
        device: Insert device (cuda or cpu). 
        writer: insert summary writer for tensorboard.
        early_stop_patience: Enter the number of epochs to wait until early stopping sets in. 

    Returns:
        results (dictionary of train and validation loss as well as all evaluation metrics from the train and validation step functions).
    """
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_f1_score = 0.0
    best_kappa = 0.0
    epochs_since_improvement = 0      
    
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_precision": [],
               "train_recall": [],
               "train_f1_score": [],
               "train_acc": [],
               "train_kappa": [],
               "val_loss": [],
               "val_precision": [],
               "val_recall": [],
               "val_f1_score": [],
               "val_acc": [],
               "val_kappa": [],
              }
    
    # Make sure to put model on target device
    model.to(device)
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_precision, train_recall, train_f1_score, train_acc, train_kappa = train_step(model=model,
                                                                                          dataloader=train_dataloader,
                                                                                          loss_fn=loss_fn,
                                                                                          optimizer=optimizer,
                                                                                          num_classes=num_classes,
                                                                                          device=device)
        
        val_loss, val_precision, val_recall, val_f1_score, val_acc, val_kappa = val_step(model=model,
                                                                                    dataloader=val_dataloader,
                                                                                    loss_fn=loss_fn,
                                                                                    num_classes=num_classes,
                                                                                    device=device)
        
        # set scheduler for decreasing learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to wandb for the current fold and epoch
        wandb.config.batch_size = batch_size
        
        wandb.log({'train_loss': train_loss, 'train_f1_score': train_f1_score,
                   'val_loss': val_loss, 'val_f1_score': val_f1_score, 'learning_rate': current_lr})
        
        # change learning rate according to the scheduler
        lr_scheduler.step()       
        
        # save model every epoch
        save_filepath = f"{experiment_num}_{model_name}_{epoch+1}_epochs.pth"
        save_model(model=model,
                       target_dir='models',
                       model_name=model_name,
                       file_name=save_filepath)
        
        # possibility to save model depending on certain conditions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} \n"
            f"Learning rate: {current_lr}\n"
            f"Train loss: {train_loss:.4f} | "
            f"Train precision: {train_precision:.4f} | "
            f"Train recall: {train_recall:.4f} | "
            f"Train f1score: {train_f1_score:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Train kappa: {train_kappa:.4f} \n"
            f"Val loss: {val_loss:.4f} | "
            f"Val precision: {val_precision:.4f} | "
            f"Val recall: {val_recall:.4f} | "
            f"Val f1score: {val_f1_score:.4f} | "
            f"Val acc: {val_acc:.4f} | " 
            f"Val kappa: {val_kappa:.4f} \n" 
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1_score"].append(train_f1_score)
        results["train_acc"].append(train_acc)
        results["train_kappa"].append(train_kappa)
        results["val_loss"].append(val_loss)
        results["val_precision"].append(val_precision)
        results["val_recall"].append(val_recall)
        results["val_f1_score"].append(val_f1_score)
        results["val_acc"].append(val_acc)
        results["val_kappa"].append(val_kappa)
        
        # Early stopping if the validation loss hasn't improved in `early_stop_patience` epochs
        if epochs_since_improvement == early_stop_patience:
            print(f"Early stopping after epoch {epoch+1}")
            break
            
    # Return the filled results at the end of the epochs
    return results


# making predictions
def make_predictions(model: torch.nn.Module, 
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_classes: int,
              device: torch.device) -> Tuple[float, float]:
    
    """Puts the trained model in evaluation mode and loops through the test dataloader.
    
    The test dataloader should ideally originate from an independent dataset unseen by the trained model.
    This ensures the most reliable evaluation of the trained model and its predictions. 

    Args:
        model: Insert previously trained model. 
        dataloader: Insert test dataloader.
        loss_fn: Insert defined loss function. 
        optimizer: Insert defined optimizer.
        num_classes: Number of classes to be classified (int). 

    Returns:
        test loss, evaluation metrics and the tree species (test_loss, test_precision, test_recall, test_f1_score, test_acc,
        test_kappa, y_preds, y_labels, species).
    """
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_precision, test_recall, test_f1_score, test_acc, test_kappa = 0, 0, 0, 0, 0, 0
    
    # set up metrics
    pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metrics = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    kappa_metrics = MulticlassCohenKappa(num_classes=num_classes, weights='quadratic').to(device)
    
    # create lists to collect predictions and labels for confusion matrix
    y_preds = []
    y_labels = []
    species_arr = []
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through the whole test DataLoader at once
        for X, y, species in tqdm(test_dataloader, desc="Making predictions"):     
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            # get prediction labels
            y_pred_class = y_pred.argmax(dim=1)
            
            # collect all labels and predictions and put them on cpu
            y_preds.append(y_pred_class.detach().cpu())
            y_labels.append(y.detach().cpu())
            species_arr.append(species)
            
            # Calculate and accumulate evaluation metrics across all batches
            pre_metrics.update(y_pred, y)
            rec_metrics.update(y_pred, y)
            f1_metrics.update(y_pred, y)
            acc_metrics.update(y_pred, y)
            kappa_metrics.update(y_pred, y)
                     
    # concatenate prediction and label tensors and covert them to numpy arrays for later use (sklearn confusion matrix)
    y_preds_tensor = torch.cat(y_preds)
    y_labels_tensor = torch.cat(y_labels)
    species_tensor = torch.cat(species_arr)
    y_preds = y_preds_tensor.numpy()
    y_labels = y_labels_tensor.numpy()
    species = species_tensor.numpy()
    
    
    # Adjust metrics to get average loss and evaluation metrics per epoch   
    test_loss = test_loss / len(test_dataloader)
    test_precision = pre_metrics.compute().item()
    test_recall = rec_metrics.compute().item()
    test_f1_score = f1_metrics.compute().item()
    test_acc = acc_metrics.compute().item()
    test_kappa = kappa_metrics.compute().item()
   
    return test_loss, test_precision, test_recall, test_f1_score, test_acc, test_kappa, y_preds, y_labels, species