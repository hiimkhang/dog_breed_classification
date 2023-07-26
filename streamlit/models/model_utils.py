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
    """
    Function to load a model from a checkpoint
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
    """
    Function to modify the last layer of a model
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
    """
    Function to save a model checkpoint
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
    """
    Function to load a model checkpoint 
    
    Return epoch, result
    """
    checkpoint = torch.load(f"{folder_path}/checkpoint_{model_type}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    helpers.log_message(f"Result in load checkpoints: {checkpoint['result']}")
    return checkpoint['epoch'] + 1, checkpoint['result']