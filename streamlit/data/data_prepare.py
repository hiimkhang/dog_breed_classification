# data/data_prepare.py

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import os

global data_dir, train_batch_size, eval_batch_size, num_workers, device

def get_data_loaders(data_dir: str, 
                     train_batch_size=32, 
                     eval_batch_size=64, 
                     num_workers=8) -> tuple[dict, dict]:
    """ Function to load the dataset and create data loaders

    Args:
        data_dir (str): dataset directory
        train_batch_size (int, optional): train batch size. Defaults to 32.
        eval_batch_size (int, optional): evaluation batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 8.

    Returns:
        tuple[dict, dict]: dataloaders and image_datasets
    """
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
        ]),
        'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: ImageFolder(root=os.path.join(data_dir, x),
                       transform=data_transforms[x])
        for x in ['train', 'valid']
    }
    
    # Define the samplers for training and evaluation batches
    sampler = {
        'train': RandomSampler(image_datasets['train']),
        'valid': SequentialSampler(image_datasets['valid'])
    }

    # Define the dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=train_batch_size if x == 'train' else eval_batch_size,
                      sampler=sampler[x],
                      num_workers=num_workers)
        for x in ['train', 'valid']
    }

    return dataloaders, image_datasets

def get_input_tensor_shape(dataloaders: dict) -> tuple[int, int, int]:
    """ Function to get the shape of the input tensor. Can be use to get
    the number of channels, height and width of the input tensor.

    Args:
        dataloaders (dict): dataloaders

    Returns:
        tuple[int, int, int]: shape of the input tensor (channels, height, width).
    """
    # Get a batch of training data
    inputs, _ = next(iter(dataloaders['train']))

    # Get the shape of the input tensor
    input_shape = inputs.shape[1:]

    return input_shape

def get_output_size(image_datasets: dict) -> int:
    """
    Function to get the size of the output tensor
    """
    # Get the size of the output tensor
    output_size = len(image_datasets['train'].classes)

    return output_size
