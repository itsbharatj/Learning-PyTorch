'''
Using Tensorboard for viz: 

    - Test Images (of a batch)
    - Model Structure Print
    - Average Loss and Accuracy of the training
    - Plot Precision-Recall Curve and ROC Curve 

- Import tensorflow for PyTorch module 
- Make a Summary Writer object
- Write the different boards to the summary writer
'''


import torch
import torch.utils
import torchvision 
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import sys
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("runs/MNIST_3")

'''
We need to make a MNIST classifier Neural Netwrok from PyTorch 

- Download the dataset from torchvision 
- Apply Transforms (Convert the images into a PyTorch Tensor)
- Store it as a PyTorch Dataloader 

- Define a NN (2 Linear Layers)
- Use Cross Entropy as the criterion 
- Start the training  (Need to resize the input as well)
'''

## How does it know not to download the data again, if it is downloaded once
## If the data is stored in a folder then how did it name reference worked? 


train_data = torchvision.datasets.MNIST(root="./Day_15_Feed_Forward_Network",train=True,download=True, 
                                        transform=torchvision.transforms.ToTensor())

print(f'Type of train_data {type(train_data)}\nTrain Data: {train_data}')

test_data = torchvision.datasets.MNIST(root="./Day_15_Feed_Forward_Network",train=False,download=False, 
                                        transform=torchvision.transforms.ToTensor())

## Hyperparameters: 
batch_size = 100 
lr = 0.001
n_epochs = 4
input_size = 784
hidden_size = 100
output_size = 10 
device = torch.device('cpu')

## Define the dataloaders: 
data_l_train = DataLoader(train_data,batch_size=batch_size,shuffle=True)
data_l_test = DataLoader(test_data,batch_size=batch_size,shuffle=True)
data_iter_train = iter(data_l_train)
data_iter_test = iter(data_l_test)

samples, labels = next(data_iter_train)
print(f'Samples Shape: {samples.shape}, Labels Shape: {labels.shape}')

## We need to plot this to see how the images come: 

# for i in range(6): 
#     plt.subplot(2,3,i+1)
#     ## Why are we using [i],[0] only --> Refers to the image, 1st index is the sample, rest is the image array 
#     plt.imshow(samples[i][0],cmap='gray')
#     plt.title(labels[i])
# # plt.show()

grid = torchvision.utils.make_grid(samples)
writer.add_image("MNIST_Batch",grid)
writer.close()
# sys.exit()
## Now we need to define a Simple NN for the classification

class Neural_Network_Classifier(nn.Module): 
    def __init__(self,input_size, hidden_size, output_size): 
        super(Neural_Network_Classifier,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x): 
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    

model = Neural_Network_Classifier(input_size=input_size,hidden_size=hidden_size,output_size=output_size).to(device)
criterion = nn.CrossEntropyLoss()
train_data_size = train_data.__len__()
n_itter = int(train_data_size/batch_size)

writer.add_graph(model=model,input_to_model=samples.reshape(-1,28*28))
# sys.exit()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

## training loop: 
'''

So the model expect the input of 784 values, 
we first have to resize the input. 

But it has a sample of 100 images, we need to process in parallel (100 NN) and then computing the total batch loss and updating the weights

How will it compute at parallel --> That can happen I think automatically? 

'''
start = time.time()
running_loss = 0.0 
running_corrects = 0

for epoch in range(n_epochs): 
    ## Ind is the batch number, one batch has 100 images and labels in the data tupple
    ## One batch is one itteration 
    ## data_size/batch ==> One Epoch
    for ind,data in enumerate(data_l_train): 
        
        ##Reshape 
        images, labels = data
        images = images.reshape(-1,784).to(device)
        labels = labels.to(device)
        ## Forward:
        y_pred = model(images)
        loss = criterion(y_pred,labels)

        ##Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Calculating the running loss and running correct! 
        running_loss += loss.item()

        ## Get the predictions from the raw values: 
        _,preds = torch.max(y_pred,1)
        running_corrects = (preds == labels).sum().item()

        if(ind%50 ==0): 
            print(f'Epoch: {epoch}, Itteration: {ind}, Loss: {loss}\nav. loss: {running_loss/50},av. accuracy: {running_corrects/50}')
            writer.add_scalar("Average Loss over 100 itterations",running_loss/50, (epoch*batch_size + ind)) 
            ## This gives an average number of correct predictions over 100 itterations 
            writer.add_scalar("Average Corrects over 100 itterations",(running_corrects/50), (epoch*batch_size + ind))
            writer.close()
            running_loss = 0.0 
            running_corrects = 0 

sys.exit()
end = time.time() - start  
print(end)


with torch.no_grad(): 

    ## Printing the test examples on the trained model: 
    '''
    We have the model, we need to make the predictions using the model: 
    '''

    n_itter_test = int(test_data.__len__()/batch_size)


    data = iter(data_l_test)
    data = next(data)
    images,labels = data
    images2 = images.reshape(-1,784)
    y_pred = model(images2)
    print(y_pred.shape)
    _,predictions = torch.max(y_pred,1)

    n_correct = (predictions==y_pred).sum().item()
    n_samples = labels.shape[0]


    for i in range(20): 
        plt.subplot(5,4,i+1)
        plt.imshow(images[i][0],cmap='grey')
        plt.title(f'{labels[i].item()},{predictions[i]}')
    # plt.show()
    plt.savefig('./Day_15_Feed_Forward_Network/test_plots.png')
    loss = criterion(y_pred,labels)

    print(f"Loss {loss}")