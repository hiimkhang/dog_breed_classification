# Dog Breed Classification

A python program that can load a dog breed dataset, applying transfer learning techniques that allow users to load their pretrained model to train a convolutional neural network to classify dog breeds. 

Coming soon: classify dog breeds from user-supplied images.

## Table of Contents
- [Dog Breed Classification](#dog-breed-classification)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Command-line Arguments](#command-line-arguments)
  - [Example](#example)

## Dataset

The dataset is taken from this [Kaggle's Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).

Consisiting of 120 classes of dog breeds, the original dataset is split into 2 folders: train and validation. Each folder contains 120 subfolders, each corresponding to a different dog breed. Each subfolder contains 100-150 images of that specific dog breed.

To make it easier to test, I made three subsets from the original dataset:
- `dog_breed_3`: 3 classes of dog breeds.
- `dog_breed_10`: 10 classes of dog breeds.
- `dog_breed_120`: 120 classes of dog breeds, the full dataset.

## Installation

1. Clone this repository to your local machine.
2. The `python_code` folder is for the python program only, he `streamlit` folder is for the streamlit app. Read README.md in those for more information.
3. Install the required dependencies by running:
   
```
pip install -r requirements.txt
```


## Usage

Run the main script to start training the model and generate loss and accuracy curves.


### Command-line Arguments

The following command-line arguments are available:

- `--arch`: Model architecture (default: "resnet50").
- `--dataset`: Path to the dataset (default: "../dog_breed_3").
- `--epochs`: Number of epochs for training (default: 10).

Additional optional arguments:

- `-r`, `--resume_training`: Resume training from a checkpoint (default: False).
- `-h`, `--help`: Show help message and exit.

## Example

To train the model using the default settings, run the following command:


