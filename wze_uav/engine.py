import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter
from wze_uav.utils2 import *


# train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               #scheduler: torch.optim.lr_scheduler,
               num_classes: int,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()
    
    # Setup train loss and other evaluation metrics
    train_loss, train_precision, train_recall, train_f1_score = 0, 0, 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate, define weights to account for imbalance and accumulate balanced loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        #loss_weighted = loss_fn(y_pred, y)
        #train_loss_weighted += loss_weighted.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()
        #loss_weighted.backward()

        # 5. Optimizer step
        optimizer.step()
        
        # 6. set scheduler for aligned lr
        #scheduler.step()
        
        # Calculate and accumulate accuracy metric across all batches
        pre_metrics = MulticlassPrecision(num_classes=num_classes, average='weighted').to(device)
        train_precision += pre_metrics(y_pred, y)
        
        rec_metrics = MulticlassRecall(num_classes=num_classes, average='weighted').to(device)
        train_recall += rec_metrics(y_pred, y)
        
        f1_metrics = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)
        train_f1_score += f1_metrics(y_pred, y)
          
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_precision = train_precision / len(dataloader)
    train_recall = train_recall / len(dataloader)
    train_f1_score = train_f1_score / len(dataloader)
    
    return train_loss, train_precision, train_recall, train_f1_score

#%%
# validation step

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_classes: int,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 
    
    # Setup validation loss and validation accuracy values
    val_loss, val_precision, val_recall, val_f1_score = 0, 0, 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            # Calculate and accumulate evaluation metrics            
            pre_metrics = MulticlassPrecision(num_classes=num_classes, average='weighted').to(device)
            val_precision += pre_metrics(y_pred, y)
            
            rec_metrics = MulticlassRecall(num_classes=num_classes, average='weighted').to(device)
            val_recall += rec_metrics(y_pred, y)
            
            f1_metrics = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)
            val_f1_score += f1_metrics(y_pred, y)
             
    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_precision = val_precision / len(dataloader)
    val_recall = val_recall / len(dataloader)
    val_f1_score = val_f1_score / len(dataloader)
   
    return val_loss, val_recall, val_precision, val_f1_score


# test step

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_classes: int,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_precision, test_recall, test_f1_score = 0, 0, 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            #loss_weighted = loss_fn(y_pred, y)
            #test_loss_weighted += loss_weighted.item() 
            
            # Calculate and accumulate evaluation metrics            
            pre_metrics = MulticlassPrecision(num_classes=num_classes, average='weighted').to(device)
            test_precision += pre_metrics(y_pred, y)
            
            rec_metrics = MulticlassRecall(num_classes=num_classes, average='weighted').to(device)
            test_recall += rec_metrics(y_pred, y)
            
            f1_metrics = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)
            test_f1_score += f1_metrics(y_pred, y)
             
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_precision = test_precision / len(dataloader)
    test_recall = test_recall / len(dataloader)
    test_f1_score = test_f1_score / len(dataloader)
    
   
   
    return test_loss, test_recall, test_precision, test_f1_score


# define train()
# Create a writer with all default settings
writer = SummaryWriter()

def train(model: torch.nn.Module,
          n_bands: int,
          batch_size: int,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          #scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          num_classes: int,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    
   
        
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_precision": [],
               "train_recall": [],
               "train_f1_score": [],
               "val_loss": [],
               "val_precision": [],
               "val_recall": [],
               "val_f1_score": [] 
              }
    
    # Make sure to put model on target device
    model.to(device)
    
    # 3. Loop through training and testing steps for a number of epochs
    #_weighted, train_precision, train_recall, train_f1_score 
    for epoch in tqdm(range(epochs)):
        train_loss, train_precision, train_recall, train_f1_score = train_step(model=model,
                                                                                          dataloader=train_dataloader,
                                                                                          loss_fn=loss_fn,
                                                                                          optimizer=optimizer,
                                                                                          #scheduler=scheduler,
                                                                                          num_classes=num_classes,
                                                                                          device=device)
        
        val_loss, val_precision, val_recall, val_f1_score = val_step(model=model,
                                                                                    dataloader=val_dataloader,
                                                                                    loss_fn=loss_fn,
                                                                                    num_classes=num_classes,
                                                                                    device=device)
        
        save_filepath = f"01_{epoch}_epochs.pth"
        save_model(model=model,
                   target_dir="models",
                   model_name=save_filepath)
        
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} \n"
            f"Train loss: {train_loss:.4f} | "
            f"Train precision: {train_precision:.4f} | "
            f"Train recall: {train_recall:.4f} | "
            f"Train f1score: {train_f1_score:.4f} \n"
            f"Val loss: {val_loss:.4f} | "
            f"Val precision: {val_precision:.4f} | "
            f"Val recall: {val_recall:.4f} | "
            f"Val f1score: {val_f1_score:.4f} \n"     
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1_score"].append(train_f1_score)
        results["val_loss"].append(val_loss)
        results["val_precision"].append(val_precision)
        results["val_recall"].append(val_recall)
        results["val_f1_score"].append(val_f1_score)
        
        # 6. Add evaluation results to SummaryWriter
        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)

            writer.add_scalars(main_tag="Precision",
                               tag_scalar_dict={"train_precision": train_precision,
                                                "val_precision": val_precision},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="Recall",
                               tag_scalar_dict={"train_recall": train_recall,
                                                "val_recall": val_recall},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="F1Score",
                               tag_scalar_dict={"train_f1_score": train_f1_score,
                                                "val_f1_score": val_f1_score},
                               global_step=epoch)
            
        # Track the PyTorch model architecture
            writer.add_graph(model=model,
                             # Pass in an example input
                             input_to_model=torch.randn(batch_size, n_bands, 250, 250).to(device))
    
    # Close the writer
            writer.close()
      
        else:
            pass
    # 6. Return the filled results at the end of the epochs
    return results
