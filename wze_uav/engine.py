import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelConfusionMatrix
from torch.utils.tensorboard import SummaryWriter


# train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               #scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0 #train_precision, train_recall, train_f1_score = 0, 0, 0, 0 # train_loss, train_precision, train_recall, train_f1_score, train_acc
    
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
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
               
        #pre_metrics = MultilabelPrecision(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
        #train_precision += pre_metrics(y_pred, y)
        
        #rec_metrics = MultilabelRecall(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
        #train_recall += rec_metrics(y_pred, y)
        
        #f1_metrics = MultilabelF1Score(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
        #train_f1_score += f1_metrics(y_pred, y)
          
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc #train_precision, train_recall, train_f1_score train_loss, train_precision, train_recall, train_f1_score

#%%
# test step

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0 #test_precision_micro, test_precision_macro, test_precision_weighted, test_precision_label, test_recall_micro, test_recall_macro, test_recall_weighted, test_recall_label, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted, test_f1_score_label = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
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
            
            # Calculate and accumulate accuracy
            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            ## Calculate and accumulate precision
            ##micro precision
            #pre_metrics_micro = MultilabelPrecision(treshold=0.5, num_labels=len(y_pred[0]), average='micro').to(device)
            #test_precision_micro += pre_metrics_micro(y_pred, y)
            ##macro precision
            #pre_metrics_macro = MultilabelPrecision(treshold=0.5, num_labels=len(y_pred[0]), average='macro').to(device)
            #test_precision_macro += pre_metrics_macro(y_pred, y)
            ##weighted precision
            #pre_metrics_weighted = MultilabelPrecision(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
            #test_precision_weighted += pre_metrics_weighted(y_pred, y)
            ## precision per label
            #pre_metrics_label = MultilabelPrecision(treshold=0.5, num_labels=len(y_pred[0]), average='none').to(device)
            #test_precision_label += pre_metrics_label(y_pred, y)
            
            ##Calculate and accumulate recall
            ##micro recall
            #rec_metrics_micro = MultilabelRecall(treshold=0.5, num_labels=len(y_pred[0]), average='micro').to(device)
            #test_recall_micro += rec_metrics_micro(y_pred, y)
            ##macro recall
            #rec_metrics_macro = MultilabelRecall(treshold=0.5, num_labels=len(y_pred[0]), average='macro').to(device)
            #test_recall_macro += rec_metrics_macro(y_pred, y)
            ##weighted recall
            #rec_metrics_weighted = MultilabelRecall(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
            #test_recall_weighted += rec_metrics_weighted(y_pred, y)
            ## recall per label
            #rec_metrics_label = MultilabelRecall(treshold=0.5, num_labels=len(y_pred[0]), average='none').to(device)
            #test_recall_label += rec_metrics_label(y_pred, y)
            #
            ##Calculate and accumulate f1 score
            ##micro f1
            #f1_metrics_micro = MultilabelF1Score(treshold=0.5, num_labels=len(y_pred[0]), average='micro').to(device)
            #test_f1_score_micro += f1_metrics_micro(y_pred, y)
            ##macro f1
            #f1_metrics_macro = MultilabelF1Score(treshold=0.5, num_labels=len(y_pred[0]), average='macro').to(device)
            #test_f1_score_macro += f1_metrics_macro(y_pred, y)
            ##weighted f1
            #f1_metrics_weighted = MultilabelF1Score(treshold=0.5, num_labels=len(y_pred[0]), average='weighted').to(device)
            #test_f1_score_weighted += f1_metrics_weighted(y_pred, y)
            ## f1 score per label
            #f1_metrics_label = MultilabelF1Score(treshold=0.5, num_labels=len(y_pred[0]), average='none').to(device)
            #test_f1_score_label += f1_metrics_label(y_pred, y)
            
            #conf_metrics = MultilabelConfusionMatrix(treshold=0.5, num_labels=len(y_pred[0])).to(device)
            #conf_mat += conf_metrics(y_pred, y.int())
             
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    #test_loss_weighted = test_loss_weighted / len(dataloader)
    #
    #test_precision_micro = test_precision_micro / len(dataloader)
    #test_precision_macro = test_precision_macro / len(dataloader)
    #test_precision_weighted = test_precision_weighted / len(dataloader)
    #test_precision_label = test_precision_label / len(dataloader)
    #
    #test_recall_micro = test_recall_micro / len(dataloader)
    #test_recall_macro = test_recall_macro / len(dataloader)
    #test_recall_weighted = test_recall_weighted / len(dataloader)
    #test_recall_label = test_recall_label / len(dataloader)
    #
    #test_f1_score_micro = test_f1_score_micro / len(dataloader)
    #test_f1_score_macro = test_f1_score_macro / len(dataloader)
    #test_f1_score_weighted = test_f1_score_weighted / len(dataloader)
    #test_f1_score_label = test_f1_score_label / len(dataloader)
   
    return test_loss, test_acc# test_precision_micro, test_precision_macro, test_precision_weighted, test_precision_label, test_recall_micro, test_recall_macro, test_recall_weighted, test_recall_label, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted, test_f1_score_label


# define train()
# Create a writer with all default settings
writer = SummaryWriter()

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          #scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    
   
        
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": []
              }
    
    #results = {"train_loss_weighted": [],
    #           "train_precision": [],
    #           "train_recall": [],
    #           "train_f1_score": [],
    #           "test_loss_weighted": [],
    #           "test_precision_micro": [],
    #           "test_precision_macro": [],
    #           "test_precision_weighted": [],
    #           "test_precision_label": [],
    #           "test_recall_micro": [],
    #           "test_recall_macro": [],
    #           "test_recall_weighted": [],
    #           "test_recall_label": [],
    #           "test_f1_score_micro": [],
    #           "test_f1_score_macro": [],
    #           "test_f1_score_weighted": [],
    #           "test_f1_score_label": []
    #}
    #
    # Make sure to put model on target device
    model.to(device)
    
    # 3. Loop through training and testing steps for a number of epochs
    #_weighted, train_precision, train_recall, train_f1_score 
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           #scheduler=scheduler,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        #test_loss_weighted, test_precision_micro, test_precision_macro, test_precision_weighted, test_precision_label, test_recall_micro, test_recall_macro, test_recall_weighted, test_recall_label, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted, test_f1_score_label = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} \n"
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_acc:.4f} \n"
            #f"Train precision: {train_precision:.4f} | "
            #f"Train recall: {train_recall:.4f} | "
            #f"Train f1_score: {train_f1_score:.4f} \n"
            #f"Test loss (weighted): {test_loss_weighted:.4f} | "
            #f"Test precision: {test_precision_weighted:.4f} | "
            #f"Test recall: {test_recall_weighted:.4f} | "
            #f"Test f1_score: {test_f1_score_weighted:.4f} \n"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)
        
        #results["train_loss_weighted"].append(train_loss_weighted)
        #results["train_precision"].append(train_precision)
        #results["train_recall"].append(train_recall)
        #results["train_f1_score"].append(train_f1_score)
        #results["test_loss_weighted"].append(test_loss_weighted)
        #results["test_precision_micro"].append(test_precision_micro)
        #results["test_precision_macro"].append(test_precision_macro)
        #results["test_precision_weighted"].append(test_precision_weighted)
        #results["test_precision_label"].append(test_precision_label)
        #results["test_recall_micro"].append(test_recall_micro)
        #results["test_recall_macro"].append(test_recall_macro)
        #results["test_recall_weighted"].append(test_recall_weighted)
        #results["test_recall_label"].append(test_recall_label)
        #results["test_f1_score_micro"].append(test_f1_score_micro)
        #results["test_f1_score_macro"].append(test_f1_score_macro)
        #results["test_f1_score_weighted"].append(test_f1_score_weighted)
        #results["test_f1_score_label"].append(test_f1_score_label)
        
        # 6. Add loss results to SummaryWriter
        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                               "test_loss": test_loss},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_accuracy": train_acc,
                                               "test_accuracy": test_acc},
                               global_step=epoch)

        ## Add precision results to SummaryWriter
        #    writer.add_scalars(main_tag="Precision",
        #                       tag_scalar_dict={"train_precision": train_precision,
        #                                       "test_precision": test_precision_weighted},
        #                       global_step=epoch)
        #    
        #    writer.add_scalars(main_tag="Recall",
        #                       tag_scalar_dict={"train_recall": train_recall,
        #                                       "test_recall": test_recall_weighted},
        #                       global_step=epoch)
        #    
        #    writer.add_scalars(main_tag="F1Score",
        #                       tag_scalar_dict={"train_f1_score": train_f1_score,
        #                                       "test_f1_score": test_f1_score_weighted},
        #                       global_step=epoch)
        #    
             
        
        # Track the PyTorch model architecture
            writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(16, 3, 250, 250).to(device))
    
    # Close the writer
            writer.close()
      
        else:
            pass
    # 6. Return the filled results at the end of the epochs
    return results
