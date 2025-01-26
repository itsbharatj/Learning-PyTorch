import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
The main aim is to use the pytorch dataset and dataloader class to get the data 

Dataset --> Use to define the dataset 
Dataloader --> Use to break the data into batches, include shuffle and other things. The dataset should be the dataset class from PyTorch

Dataset: Titanic Dataset
'''

class Wine_Dataset(Dataset): 
    def __init__(self): 

        self.data = np.loadtxt(fname="wine.csv", delimiter=',',dtype=np.float32,skiprows=1)
        self.shape = self.data.shape
        self.samples, self.features = self.shape[0],self.shape[1]
        self.x = self.data[:,1:]
        self.y = self.data[:,[0]] 

        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

        self.length = self.samples
    def __getitem__(self, index): 
        return self.x[index],self.y[index]

    def __len__(self): 
        return self.length
    

if __name__ == "__main__": 
    dataset = Wine_Dataset()
    #This works! 
    print(dataset[77])

    ## DataLoader for itterations of the class: 

    loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True)

    dataiter = iter(loader)
    data_iter = next(dataiter)
    features, labels  =  data_iter

    print(features, labels)

    ## For loop to understand epochs, number of itterations and batch size. Need to print all the data --> No. of epochs 

    n_epochs = 10
    n_itterations = int(np.ceil(dataset.length/10))
    print(n_itterations)
    

    ## This is working as well
    for i in range(n_epochs): 
        for ind,j in enumerate(loader): 
            print(f'Epoch {i}, Itteration: {ind}, Data_X: {j[0]}, Data_Y: {j[1]}')