import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassCohenKappa
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from wze_uav.utils2 import *
import numpy as np


# train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               num_classes: int,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()
    
    
    # Setup train loss and other evaluation metrics
    train_loss, train_precision, train_recall, train_f1_score, train_acc = 0, 0, 0, 0, 0
    
    # set up metrics
    #pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    #rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    #f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    #kappa_metrics = MulticlassCohenKappa(num_classes=num_classes).to(device)
    labels = np.array([0,1,2])
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
        
        # get the class predictions
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Calculate and accumulate accuracy metric across all batches
        train_acc += ((y_pred_class == y).sum().item()/len(y_pred_class))
        y_pred_class = y_pred_class.detach().cpu().numpy() 
        y = y.detach().cpu().numpy()
        train_precision += precision_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
        train_recall += recall_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
        train_f1_score += f1_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
        
        
        #train_precision += pre_metrics(y_pred_class, y_class)
        #train_recall += rec_metrics(y_pred_class, y_class)
        #train_f1_score += f1_metrics(y_pred_class, y_class)
        #train_kappa += kappa_metrics(y_pred_class,y_class)
        #
        #y_pred = y_pred.argmax(dim=1).detach().cpu().numpy() 
        #y = y.detach().cpu().numpy()
        #train_precision += precision_score(y, y_pred, average='macro', zero_division=1)
        #train_recall += recall_score(y, y_pred, average='macro', zero_division=1)
        #train_f1_score += f1_score(y, y_pred, average='macro', zero_division=1)
          
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_precision = train_precision / len(dataloader)
    train_recall = train_recall / len(dataloader)
    train_f1_score = train_f1_score / len(dataloader)
    train_acc = train_acc / len(dataloader)
   
    return train_loss, train_precision, train_recall, train_f1_score, train_acc

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
    val_loss, val_precision, val_recall, val_f1_score, val_acc = 0, 0, 0, 0, 0
    
    #pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    #rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    #f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    #kappa_metrics = MulticlassCohenKappa(num_classes=num_classes).to(device)
    labels = np.array([0,1,2])
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()
            
            # get predicted classes
            y_pred_class = y_pred_logits.argmax(dim=1)
            
            # Calculate and accumulate evaluation metrics
            val_acc += ((y_pred_class == y).sum().item()/len(y_pred_class))
            y_pred_class = y_pred_class.detach().cpu().numpy() 
            y = y.detach().cpu().numpy()
            val_precision += precision_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
            val_recall += recall_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
            val_f1_score += f1_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2]) # if None -> , labels=labels (labels=np.array([0,1,2])
            
                        
            #val_precision += pre_metrics(y_pred_class, y_class)
            ##
            #val_recall += rec_metrics(y_pred_class, y_class)
            ##
            #val_f1_score += f1_metrics(y_pred_class, y_class)
            ##
            #val_kappa += kappa_metrics(y_pred_class, y_class)
            
            #y_pred = y_pred.argmax(dim=1).detach().cpu().numpy() 
            #y = y.detach().cpu().numpy()
            #val_precision += precision_score(y, y_pred, average='macro', zero_division=1)
            #val_recall += recall_score(y, y_pred, average='macro', zero_division=1)
            #val_f1_score += f1_score(y, y_pred, average='macro', zero_division=1)
             
    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_precision = val_precision / len(dataloader)
    val_recall = val_recall / len(dataloader)
    val_f1_score = val_f1_score / len(dataloader)
    val_acc = val_acc / len(dataloader)
    
   
    return val_loss, val_recall, val_precision, val_f1_score, val_acc


# test step

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_classes: int,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_precision, test_recall, test_f1_score, test_acc = 0, 0, 0, 0, 0
    
    #pre_metrics = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    #rec_metrics = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    #f1_metrics = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    #kappa_metrics = MulticlassCohenKappa(num_classes=num_classes).to(device)
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()
            
            # get predicted classes
            y_pred_class = y_pred_logits.argmax(dim=1)
            
            # Calculate and accumulate evaluation metrics    
            test_acc += ((y_pred_class == y).sum().item()/len(y_pred_class))
            y_pred_class = y_pred_class.detach().cpu().numpy() 
            y = y.detach().cpu().numpy()
            test_precision += precision_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
            test_recall += recall_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
            test_f1_score += f1_score(y, y_pred_class, average='macro', zero_division=0, labels=[0,1,2])
            
            #test_precision += pre_metrics(y_pred_class, y_class)
            ##
            #test_recall += rec_metrics(y_pred_class, y_class)
            ##
            #test_f1_score += f1_metrics(y_pred_class, y_class)
            ##
            #test_kappa += kappa_metrics(y_pred_class, y_class)
            
            #y_pred = y_pred.argmax(dim=1).detach().cpu().numpy() 
            #y = y.detach().cpu().numpy()
            #test_precision += precision_score(y, y_pred, average='macro', zero_division=1)
            #test_recall += recall_score(y, y_pred, average='macro', zero_division=1)
            #test_f1_score += f1_score(y, y_pred, average='macro', zero_division=1)
             
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_precision = test_precision / len(dataloader)
    test_recall = test_recall / len(dataloader)
    test_f1_score = test_f1_score / len(dataloader)
    test_acc = test_acc / len(dataloader)
   
   
    return test_loss, test_recall, test_precision, test_f1_score, test_acc


# define train()
# Create a writer with all default settings
writer = SummaryWriter()

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
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    
   
        
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_precision": [],
               "train_recall": [],
               "train_f1_score": [],
               "train_acc": [],
               "val_loss": [],
               "val_precision": [],
               "val_recall": [],
               "val_f1_score": [],
               "val_acc": [],
              }
    
    # Make sure to put model on target device
    model.to(device)
    
    # 3. Loop through training and testing steps for a number of epochs
    #_weighted, train_precision, train_recall, train_f1_score 
    for epoch in tqdm(range(epochs)):
        train_loss, train_precision, train_recall, train_f1_score, train_acc = train_step(model=model,
                                                                                          dataloader=train_dataloader,
                                                                                          loss_fn=loss_fn,
                                                                                          optimizer=optimizer,
                                                                                          num_classes=num_classes,
                                                                                          device=device)
        
        val_loss, val_precision, val_recall, val_f1_score, val_acc = val_step(model=model,
                                                                                    dataloader=val_dataloader,
                                                                                    loss_fn=loss_fn,
                                                                                    num_classes=num_classes,
                                                                                    device=device)
        
        # set scheduler for aligned lr
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")
        lr_scheduler.step()
        
        
        save_filepath = f"01_{epoch+1}_epochs.pth"
        save_model(model=model,
                   target_dir=model_name,
                   model_name=save_filepath)
        
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} \n"
            f"Train loss: {train_loss:.4f} | "
            f"Train precision: {train_precision:.4f} | "
            f"Train recall: {train_recall:.4f} | "
            f"Train f1score: {train_f1_score:.4f} | "
            f"Train acc: {train_acc:.4f} \n"
            f"Val loss: {val_loss:.4f} | "
            f"Val precision: {val_precision:.4f} | "
            f"Val recall: {val_recall:.4f} | "
            f"Val f1score: {val_f1_score:.4f} | "
            f"Val acc: {val_acc:.4f} \n" 
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1_score"].append(train_f1_score)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_precision"].append(val_precision)
        results["val_recall"].append(val_recall)
        results["val_f1_score"].append(val_f1_score)
        results["val_acc"].append(val_acc)
        
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
            
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_accuracy": train_acc,
                                                "val_accuracy": val_acc},
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
