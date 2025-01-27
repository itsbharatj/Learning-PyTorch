import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torchvision 
import torchvision.transforms as transforms 
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import time 
from save_load_model import save_load

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

device = torch.device("mps")
## Define the Hyperparameters: 
batch_size = 100
n_epochs = 100
lr = 0.01

class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship","truck"]
## Transforms: 
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

## Download the dataset

train_data = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
test_data_normal = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=torchvision.transforms.ToTensor())


train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)
test_loader_normal = DataLoader(test_data_normal,batch_size=batch_size,shuffle=False)

model_save = save_load 

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

        out = out.view(-1,16*5*5)  ##Need to understand this for now

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


model = CNN_Model().to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=lr)
data_size = train_data.__len__()

data_eg_train = iter(train_loader)
data_eg_train = next(data_eg_train)

images, labels = data_eg_train

print(f'Shape of data tensor {images.shape}, Labels: {labels.shape}, Example Labels: {labels[:10]}')


# # # Load the saved state
# model, optim, epoch, loss = save_load.load_model(model, optim, 'cifar10_cnn.pth', device)

# # Model is now ready for inference
# model.eval()  # Set to evaluation mode

loss_epoch = {}

## Training: 
st = time.time()
for epoch in range(n_epochs): 
    for ind,(images,labels) in enumerate(train_loader): 

        images = images.to(device)
        labels = labels.to(device)
        #This is a batch: 100 images together: What is the shape of these images? --> [100,3,32,32] 
        y_pred = model(images) # --> 

        ## Loss: 
        loss = criterion(y_pred,labels)

        ## Backpropogation: 

        optim.zero_grad()   
        loss.backward()
        optim.step()

        ## Print the information of the training: 

        if (ind%200)==0: 
            print(f'Epoch: {epoch}, Itteration: {ind}, Loss: {loss}')
    
    loss_epoch[epoch] = loss.item()
    
    model_save.save_model(model, optim, epoch, loss, 'cifar10_cnn.pth')

# print(f'Time for training: {time.time()-st}')

print(loss_epoch)
plt.figure(1)
plt.plot(loss_epoch.keys(),loss_epoch.values())
plt.show()


## Print the test samples: 

with torch.no_grad(): 
    train_batch_normal = iter(test_loader_normal)
    train_batch_normal = next(train_batch_normal)
    images_normal,_ = train_batch_normal

    
    train_batch = iter(test_loader)
    train_batch = next(train_batch)

    

    images, labels = train_batch
    images = images.to(device)
    labels = labels.to(device)

    model_prediction = model(images)

    ## We need to find the max value for the final predictions: 

    _,prediction_label = torch.max(model_prediction,1)

    plt.figure(2)
    ## Example batch printing: 
    for i in range(6): 
        plt.subplot(2,3,i+1)
        plt.imshow(images_normal[i].T.cpu())
        plt.title(f'Prediction:{class_labels[prediction_label[i]]}, Actual:{class_labels[labels[i]]}')
    plt.show()

    ##Actual Accuracy of the model: 

    n_samples_iter = 0 
    n_correct = 0 
    for (images,labels) in test_loader: 

        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        
        ## Actual labels: 
        _,model_pred = torch.max(y_pred,1)

        n_correct += (model_pred==labels).sum().item()
        n_samples_iter += labels.shape[0]

    accuracy = (n_correct/n_samples_iter)*100
    print(f'Total Accuracy for all the classes: {accuracy}')


