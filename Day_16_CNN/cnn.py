import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torchvision 
import torchvision.transforms as transforms 
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt 


'''
We need to make a CNN to classify the CIFAR-10 Dataset. Contains the images and the classes for 10 objects. 

- Download the dataset 
- Define the transforms (Make the dataset into a tensor, normalize the dataset as well?)
- define the hyperparameters 
- define the CNN model: 
    - Image (3 channel)
    - Convolution Layer (6 channel, kernel_size=5)
    - ReLU 
    - Max Pooling (Kernel Size=2, Stride=2)
    - Convolution (input_channels=6,output_channels=10, kernel_size=3)
    - RelU 
    - Max Pooling(Kernel_Size=?, Stride=2)
    - Flatten ==> 
    - FC1: Input_size: , output_size: 128
    - FC2: Input_size: 128, Output_size:84 
    - FC3: Input_size: 84, Output_size: 10 (Number of classes)

- Start the training
- Make the inferences
'''

## Define the Hyperparameters: 
batch_size = 10 
n_epochs = 3 
lr = 0.001

## Transforms: 
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

## Download the dataset

train_data = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

##Define the CNN Model: 

class CNN_Model(nn.Module):
    def __init__(self,input_channels=3,conv1_c=6,conv2_c=16): 
        super(CNN_Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels = conv1_c,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_c,out_channels=conv2_c,kernel_size=5)
        self.fc1 = nn.Linear(16*5*5,128)
        self.fc2 = nn.Linear(128,82)
        self.fc3 = nn.Linear(82,10)

    def forward(self,x): 
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))

        out = out.view(-1,16*5*5)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=lr)
data_size = train_data.__len__()

data_eg_train = iter(train_loader)
data_eg_train = next(data_eg_train)

images, labels = data_eg_train

print(f'Shape of data tensor {images.shape}, Labels: {labels.shape}')

## Training: 

for epoch in range(n_epochs): 
    for ind,(images,labels) in enumerate(train_loader): 
        #This is a batch: 100 images together: What is the shape of these images? --> [100,3,32,32] 
        y_pred = model(images) # --> 

        ## Loss: 
        loss = criterion(y_pred,labels)

        ## Backpropogation: 

        optim.zero_grad()
        loss.backward()
        optim.step()

        ## Print the information of the training: 

        if (ind%10)==0: 
            print(f'Epoch: {epoch}, Itteration: {ind}, Loss: {loss}')




        


