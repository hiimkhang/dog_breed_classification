# models/model_utils.py

import torch
import torch.nn as nn
from torchvision import models
import data_prepare as dp
import os

import random
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

default_checkpoint_dir = "./checkpoints"

def get_available_modeltypes() -> list[str]:
    available_models = []
    with open("available_models.txt", "r") as f:
        for line in f.readlines():
            available_models.append(line.strip())
    
    return available_models

def load_model(
               model_type:str, 
               in_channels: int,
               num_classes: int,
               device: torch.device,
               path = "") -> nn.Module:
    """
    Function to load a model from a checkpoint
    """
        
    # If user input model path and is resume, load the state dict 
    if path:
        model = models.__dict__[model_type](pretrained=False)
        if check_pretrained_model_path(path):
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
        log_message("Unsupported model type. Only nn.Sequential or nn.Module models are supported.")

    # Change out_features of last layer
    if isinstance(last_layer, nn.Linear):
        last_layer.out_features = num_classes
    else:
        log_message("The last layer is not of type nn.Linear, and its out_features cannot be modified.")
    
    # Change in_channels of first layer
    if isinstance(first_layer, nn.Conv2d):
        first_layer.in_channels = in_channels
    else:
        log_message("The last layer is not of type nn.Linear, and its out_features cannot be modified.")
    
    return model

def save_checkpoint(model: nn.Module, 
                    model_type: str,
                    optimizer: torch.optim.Optimizer, 
                    epoch: int,
                    num_epochs: int,
                    dataset_name: str,
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
    
    check_and_create_dir(folder_path)

    if epoch >= num_epochs or len(result['Train loss']) >= num_epochs:
        if os.path.isfile(f"{folder_path}/checkpoint_{model_type}_{dataset_name}.pth"):
            os.remove(f"{folder_path}/checkpoint_{model_type}_{dataset_name}.pth")
    else:
        torch.save(checkpoint, f"{folder_path}/checkpoint_{model_type}_{dataset_name}.pth")
    
def load_checkpoint(model_type: str, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    dataset_name:str,
                    folder_path = default_checkpoint_dir):
    """
    Function to load a model checkpoint 
    
    Return epoch, result
    """
    checkpoint = torch.load(f"{folder_path}/checkpoint_{model_type}_{dataset_name}.pth")
    log_message(f"Checkpoint found! Resume training {model_type} from epoch {checkpoint['epoch'] + 1}...", is_print=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_message(f"Result in load checkpoints: {checkpoint['result']}", is_print=False)
    return checkpoint['epoch'] + 1, checkpoint['result']

def check_path(path: str) -> str:
    """
    Function to check if a path exists, and if not, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def count_folders(path: str) -> int:
    count = 0
    for item in os.listdir(path):
        if (os.path.isdir(os.path.join(path, item))):
            count += 1
    return count

def split_folders(path: str, train_percentage: float = 0.8) -> None:
    target_dir = "./data"
    
    train_folder = os.path.join(target_dir, "train")
    validation_folder = os.path.join(target_dir, "validation")
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    
    for class_name in os.listdir(path):
        class_folder = os.path.join(path, class_name)
        if os.path.isdir(class_folder):
            images = os.listdir(class_folder)
            num_images = len(images)
            num_train = int(num_images * train_percentage)
            
            random.shuffle(images)
            
            train_images = images[:num_train]
            valid_images = images[num_train:]
            
            for image in train_images:
                src_path = os.path.join(class_folder, image)
                dest_path = os.path.join(train_folder, class_name, image)
                os.makedirs(os.path.join(train_folder, class_name), exist_ok=True)
                shutil.copy(src_path, dest_path)
            
            for image in valid_images:
                src_path = os.path.join(class_folder, image)
                dest_path = os.path.join(validation_folder, class_name, image)
                os.makedirs(os.path.join(validation_folder, class_name), exist_ok=True)
                shutil.copy(src_path, dest_path)

def check_dataset_path(path: str) -> bool:
    """
    Function to check if a dataset path exists, and if not, create it
    """
    if os.path.exists(path):
        if not os.path.exists(os.path.join(path, 'train')) or not os.path.exists(os.path.join(path, 'valid')):
            # If the data folder contains multiple folders (classes), splitting is required
            if count_folders(path) > 1:
                log_message('Splitting dataset into train and validation folders into ./data/...')
                split_folders(path)
                log_message('Splitting complete! Visit ./data/ to view the split dataset.')
                return True
        elif os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'valid')):
            return True
    return False

def check_pretrained_model_path(path: str) -> bool:
    if os.path.exists(path):
        if path.endswith('.pth') or path.endswith('.pt'):
            # log_message(f"Model found! {path}")
            return True
    log_message(f"Model not found at path: {path}")
    return False

def log_message(message: str, log_file: str = "log.txt", is_print = False):
    with open(log_file, "a") as f:
        f.write(f"{message}\n")
    if is_print:
        print(message)
    
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_and_create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_loss_accuracy_curve(results: dict, output_dir: str, save_name: str):
    check_and_create_dir(output_dir)

    epochs = range(1, len(results['Train loss']) + 1)
    results_df = pd.DataFrame(results, index=epochs)
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Train loss'], label='Train Loss')
    plt.plot(results_df['Test loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{save_name}_loss_curve.png'))
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Train accuracy'], label='Train Accuracy')
    plt.plot(results_df['Test accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{save_name}_accuracy_curve.png'))
    plt.close()