# utils/helpers.py

import os
import random
import shutil
import torch
import numpy as np

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
    """ Split a folder consists multi classes (subfolders) 
    into train and validation folder with a ratio of train_percentage.
    
    The input folder structure should be:
        path/class1/xxx.png
            ...
        path/class2/xxx.png
    Args:
        path (str): path to folder
        train_percentage (float, optional): train/split ratio. Defaults to 0.8.
    """
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
    Function to check if a dataset path exists, and if not, create it.
    
    This function calls split_folders() to split the dataset into train and validation folders.
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
    """ Check if a pretrained model exists at the given path.
    """
    if os.path.exists(path):
        if path.endswith('.pth') or path.endswith('.pt'):
            # log_message(f"Model found! {path}")
            return True
    # log_message(f"Model not found at path: {path}")
    return False

def log_message(message: str, log_file: str = "log.txt"):
    with open(log_file, "a") as f:
        f.write(f"{message}\n")
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
    