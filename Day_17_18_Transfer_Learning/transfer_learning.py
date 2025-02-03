'''
So this is the module for transfer learning: 

Transfer Learning: Using a trained model on a different context
    The primary change is that the last layer (fully connected) changes, as the number of classes 
    Does that also have to trainied again? 

Data: We have the data in a data dir, that will be picked up the PyTorch dataset module

How to get the data --> Find some kaggle directory for two or three classes of data 

/data
    /train 
        /class_1
        /class_2 
    /val
        /class_1
        /class_2

Automatically gets the class labels from this structure
- What about the test?? or val? 

Elements: 
    - Define the dataset and labels for train, val and train  
    - Define the dataset transform compose layer 
        Transformation: 
            - Resize
            - Random Horizontal Flip 
            - Random Crop/Fixed Crop
            - To Tensor
            - Normalize Pixel Values (Around mean, std)
    - Define the training function 
    - Get the model from PyTorch 
    
    - 2 Approaches: 
        1. Fine Tuning: 
            - We need to change the last layer into the number of classes (add one more layer or modify??)
            - Then need to train the model again (fine tuning)
        2. Freezing the weights: 
            - Freeze all the layers, until the last layers
            - Only train the last layers 
        
'''

import torch 
import torchvision 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 

data_dir = "data/cats_dogs"
sets = ["train","val"]
n_vals = (0.5*3)
batch_size = 4 
lr = 0.001


transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(225), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=n_vals,std=n_vals)
])

transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(225), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=n_vals,std=n_vals)
])

train_data = torchvision.datasets.ImageFolder(root=data_dir,
                    transform=transforms_train)
val_data = torchvision.datasets.ImageFolder(root=data_dir,transform=transforms_val)

## Dataloaders: 
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,lr=lr)
val_loader = DataLoader(dataset=val_data,batch_size=batch_size,lr=lr)

## Will have to read this over again: 
def train():

