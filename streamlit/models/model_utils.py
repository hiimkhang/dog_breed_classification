# models/model_utils.py

import torch
import torch.nn as nn
from torchvision import models
import streamlit as st

import data.data_prepare as dp
import utils.helpers as helpers

import utils.streamlit as streamlit
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

default_checkpoint_dir = "./checkpoints"

def get_available_modeltypes() -> list[str]:
    """ Get the available model types from the available_models.txt file
    
    User can modify the available_models.txt file to add more models

    Returns:
        list[str]: list of available model types
    """
    available_models = []
    with open("models/available_models.txt", "r") as f:
        for line in f.readlines():
            available_models.append(line.strip())
    
    return available_models

@streamlit.st.cache_resource(show_spinner=True)
def load_model(path: str, 
               model_type:str, 
               in_channels: int,
               num_classes: int,
               device: torch.device) -> nn.Module:
    """ Load a model from torchvision.models or given path.

    Args:
        model_type (str): model architecture name (resnet50, resnet152, efficientnet_b32, ...).
        in_channels (int): input channels from dataset.
        num_classes (int): total of dataset classes, map with out_features in the last layer.
        device (torch.device): device to run the model.
        path (str, optional): pretrained model path. Defaults to "".

    Returns:
        nn.Module: model with modified first and last layer.
    """
        
    # If user input model path and, load the state dict 
    if streamlit.st.session_state['model_path']:
        model = models.__dict__[model_type](pretrained=False)
        if helpers.check_pretrained_model_path(path):
            model.load_state_dict(torch.load(path, map_location=device))
    else:
        model = models.__dict__[model_type](pretrained=True)
        
    model = modify_layers(model, in_channels, num_classes)
    model.to(device)
    return model

def modify_layers(model: nn.Module, 
                  in_channels: int,
                  num_classes: int) -> nn.Module:
    """ Modify the first and the last layer of the model.

    Args:
        model (nn.Module): model to modify.
        in_channels (int): input channels from dataset.
        num_classes (int): number of classes from dataset.

    Returns:
        nn.Module: model with modified first and last layer.
    """
    last_layer = None
    first_layer = None
    pos = 0

    # Get the first and last layer of the model
    if isinstance(model, nn.Sequential):
        last_layer = model[-1]
        first_layer = model[0]
    elif isinstance(model, nn.Module):
        for _, module in model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:
                pos += 1
                if pos == 1:
                    first_layer = module
                last_layer = module
    else:
        st.error("Unsupported model type. Only nn.Sequential or nn.Module models are supported.")
        helpers.log_message("Unsupported model type. Only nn.Sequential or nn.Module models are supported.")

    # Change out_features of last layer
    if isinstance(last_layer, nn.Linear):
        last_layer.out_features = num_classes
    else:
        st.error("The last layer is not of type nn.Linear, and its out_features cannot be modified.")
        helpers.log_message("The last layer is not of type nn.Linear, and its out_features cannot be modified.")
    
    # Change in_channels of first layer
    if isinstance(first_layer, nn.Conv2d):
        first_layer.in_channels = in_channels
    else:
        st.error("The first layer is not of type nn.Conv2d, and its in_channels cannot be modified.")
        helpers.log_message("The last layer is not of type nn.Linear, and its out_features cannot be modified.")
    
    return model

def save_checkpoint(model: nn.Module, 
                    model_type: str,
                    optimizer: torch.optim.Optimizer, 
                    epoch: int,
                    num_epochs: int,
                    result: dict[str, list],
                    folder_path = default_checkpoint_dir) -> None:
    """ Save a model checkpoint, including epoch, optimizer, model state dict and result.

    Args:
        model (nn.Module):  model
        model_type (str):  model architecture name (resnet50, resnet152, efficientnet_b32, ...).
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): continue from this epoch.
        num_epochs (int): total epochs.
        dataset_name (str): to name the checkpoint .pth file, e.g. checkpoint_resnet50_cifar10.pth
        result (dict[str, list]): the loss and accuracy up until the current epoch.
        folder_path (_type_, optional): defaults to default_checkpoint_dir.
    """
    checkpoint = {
        'epoch': epoch,
        'num_epochs': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'result': result
    }
    
    helpers.check_and_create_dir(folder_path)

    if epoch >= num_epochs or len(result['Train loss']) >= num_epochs:
        os.remove(f"{folder_path}/checkpoint_{model_type}.pth")
    else:
        torch.save(checkpoint, f"{folder_path}/checkpoint_{model_type}.pth")
    
def load_checkpoint(model_type: str, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    folder_path = default_checkpoint_dir):
    """ Load a model checkpoint, including epoch, optimizer, model state dict and result.

    Args:
        model_type (str): model architecture name (resnet50, resnet152, efficientnet_b32, ...).
        model (nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        dataset_name (str): to name the checkpoint .pth file, e.g. checkpoint_resnet50_cifar10.pth
        folder_path (_type_, optional): defaults to default_checkpoint_dir.

    Returns:
        tuple(int, dict): continue epoch (int) and result (dict)
    """
    checkpoint = torch.load(f"{folder_path}/checkpoint_{model_type}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    helpers.log_message(f"Result in load checkpoints: {checkpoint['result']}")
    return checkpoint['epoch'] + 1, checkpoint['result']