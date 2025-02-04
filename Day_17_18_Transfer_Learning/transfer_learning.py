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
import time
import copy
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np

data_dir = "data/cats_dogs"
sets = ["train","val"]
n_vals = {
    "mean":[0.485, 0.456, 0.406], 
    "std": [0.229, 0.224, 0.225]
}
batch_size = 4 
lr = 0.001

device = "mps" if torch.backends.mps.is_available() else "cpu"

transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(225), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=n_vals["mean"],std=n_vals["std"])
])

transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(225), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=n_vals["mean"],std=n_vals["std"])
])

train_data = torchvision.datasets.ImageFolder(root=f'{data_dir}/train',
                    transform=transforms_train)
val_data = torchvision.datasets.ImageFolder(root=f'{data_dir}/val',transform=transforms_val)

## Used a dict for this
Dataloaders = { 
    "train": DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True),
    "val":  DataLoader(dataset=val_data,batch_size=batch_size)
} 

dataset_size = {
    "train": len(train_data), 
    "val": len(val_data)
}

## Will have to read this over again:
'''
Read and go through: 
'''
def train(model,criterion,scheduler,optimizer,n_epochs=10): 
    '''
    We have a train and a validation dataloaders
    We have to use the train dataset to update the weights
    Valid to see what is the loss that we are getting 
    In case the accuracy is better than the last one we need to update the weights

    - make a deep copy of the weights of the model
    - init the best_acc
    - main training loop: 
        - loop for train and val dataset 
            - Set the model to train or val 
            - Itterate though the dataloader of train and val 
                - Get the output predictions the inputs 
                - Use max to get the actual predictions 
                - compute the loss 

                - if the mode is train, then: 
                    - backprop loss 
                    - clear optimizer 
                    - update the weights using otptimizer 
                
                - Compute the model accuracy: 
                    - Calculate the total accuracy 
                    - How many did you got correct 

                - Do scheduler step if it is training mode 
                - If val: 
                    - then make 
    '''

    since = time.time()

    ## Making a copy of the model weights: 
    best_model_wts  = copy.deepcopy(model.state_dict())
    best_acc = 0.0 
    model = model.to(device)

    for epoch in range(n_epochs): 
        
        for phase in ["train","val"]: 
            print(f"Epoch: {epoch}, Phase: {phase}")
            if phase == "train": 
                model.train()
            else: 
                model.eval()
            
            running_loss = 0.0 
            running_corrects = 0

            ## Itterate over the data: 
            for inputs, labels in Dataloaders[phase]: 
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase=="train"): 

                    output = model(inputs)
                    _,probs = torch.max(output,1)
                    loss = criterion(output, labels)
                    

                    ## backward step only if in the training phase: 

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    
                    ## loss.item() gives 
                    ## I do not understand this piece of code and will this even work? 

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(probs == labels)

                    epoch_loss = running_loss / dataset_size[phase]
                    epoch_acc = running_corrects.float() / dataset_size[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    if phase == "val" and epoch_acc > best_acc: 
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    print()
            
            if phase == "train": 
                scheduler.step()
    
    time_elapse = time.time()- since
    print(f"Training Complete in {time_elapse//60}")
    print(f"best acc: {best_acc}")

    model.load_state_dict(best_model_wts)
    return model

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


## This the fine tuning method:: 

##Get the features in last layers of the last layer
num_fts = model.fc.in_features
## Adding a new layer, with 2 features for own dataset
model.fc = nn.Linear(num_fts,2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

## Every 7th epoch (step_size), lr is multipled by 0.01 (gamma)
step_b_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.01)

# model = train(model=model,criterion=criterion,scheduler=step_b_scheduler,optimizer=optimizer,n_epochs=1)

## I need to see the images and labels 


# Get one batch of training data
data_train_iter = iter(Dataloaders["train"])
images, labels = next(data_train_iter)

print(f'''
        Shape of Images: {images.shape}\nImages: {images}\n
        Shape of Labels: {labels.shape}\nLabels: {labels}      
      ''')

label_names = ["cats", "dogs"]
print(' '.join(f'{label_names[labels[j]]}' for j in range(len(labels))))

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Make a grid from batch
out = torchvision.utils.make_grid(images)

imshow(out)
    

