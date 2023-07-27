import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model_utils as mu
import data_prepare as dp
from tqdm import tqdm

import pandas as pd

def train_step(model: nn.Module,
               train_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int) -> tuple[float, float]:
    """ Performs a single training step through the train loader.
    i.e. forward pass, backward pass and parameter update.

    Args:
        model (nn.Module): _model
        train_loader (DataLoader): _train loader_
        loss_fn (nn.Module): loss function, e.g. nn.CrossEntropyLoss()
        optimizer (torch.optim.Optimizer): optimizer, e.g. torch.optim.SGD()
        device (torch.device): using device
        epoch (int): current epoch

    Returns:
        tuple[float, float]: average train loss and train accuracy
    """
    model.train()
    
    train_loss, train_accuracy = 0., 0.
    
    for batch, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X) # forward pass
        
        loss = loss_fn(y_pred, y) # compute loss
        train_loss += loss.item()
        
        optimizer.zero_grad() # reset gradients
        
        loss.backward() # backward pass
        
        optimizer.step() # update parameters
        
        y_pred_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1) # compute accuracy
        train_accuracy += (y_pred_labels == y).sum().item() / len(y_pred)
    
    return train_loss/len(train_loader) , train_accuracy/len(train_loader)

def eval_step(model: nn.Module,
              test_loader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device,
              epoch: int) -> tuple[float, float]:  
     
    """ Performs a single testing step through the test loader.
    i.e. forward pass and compute loss and accuracy.

    Args:
        model (nn.Module): _model
        test_loader (DataLoader): _test loader_
        loss_fn (nn.Module): loss function, e.g. nn.CrossEntropyLoss()
        device (torch.device): using device
        epoch (int): current epoch

    Returns:
        tuple[float, float]: average test loss and test accuracy
    """
    model.eval()
    
    test_loss, test_accuracy = 0., 0.
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            
            y_pred_logits = model(X) # forward pass
            
            loss = loss_fn(y_pred_logits, y) # compute loss
            test_loss += loss.item()
            
            y_pred_labels = y_pred_logits.argmax(dim=1) # compute accuracy
            test_accuracy += (y_pred_labels == y).sum().item() / len(y_pred_labels)
    
    return test_loss/len(test_loader), test_accuracy/len(test_loader)

def train_model(model: nn.Module,
                model_type: str,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          start_epoch: int,
          epochs: int,
          device: torch.device,
          results,
          dataset_name:str):
    """ Trains the model for the given number of epochs.

    Returns:
        dict[str, list]: results of the training and testing process
    """
    model.to(device)
    
    for epoch in range(start_epoch, epochs):
                
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device, epoch)
        test_loss, test_acc = eval_step(model, test_dataloader, loss_fn, device, epoch)        
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss: {train_loss:.4f} | "
              f"Train accuracy: {train_acc:.4f} | " 
              f"Test loss: {test_loss:.4f} | "
              f"Test accuracy: {test_acc:.4f}\n")
                

        results['Train loss'].append(train_loss)
        results['Train accuracy'].append(train_acc)
        results['Test loss'].append(test_loss)
        results['Test accuracy'].append(test_acc)
        
        mu.save_checkpoint(model=model, 
                           model_type=model_type,
                           optimizer=optimizer, 
                           epoch=epoch,
                           num_epochs = epochs,
                           dataset_name=dataset_name,
                           result=results)

        
            
    return results