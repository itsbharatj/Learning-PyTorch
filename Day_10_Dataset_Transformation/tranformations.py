'''
We have a PyTorch Dataset, which we need to get some transformations applied. 

For this: 

1. Our Dataset Class: 
    - Add the transforms feature

2. Make the tranform classes: 
    - For Multiplication 
    - For Addition

3. Apply these tranformations and check it with the output that we have recieved

4. Apply a combined tranformation using the composed method of torchvision

5. Check that as well with the output that we have recieved 
'''

import sys
import os
import torch 
import torchvision

# from Day_09_Datasets_and_Dataloaders import dataloaders_datastes
from dataloaders_datastes import Wine_Dataset

class Tranform_Wine_Dataset(Wine_Dataset): 
    def __init__(self,transforms=None,*args,**kwargs): 
        super(Tranform_Wine_Dataset,self).__init__(*args,**kwargs)

        self.transforms = transforms

    def __getitem__(self, index):
        sample =  super().__getitem__(index)

        if self.transforms: 
            return self.transforms(sample)
        
        return sample

class Mul_Tranform: 
    def __init__(self,factor=2): 
        self.factor = factor
    
    def __call__(self,sample):
        inputs,targets = sample
        inputs *= self.factor
        return inputs,targets

class Add_Tranform: 
    def __init__(self,term=10): 
        self.add_term = term
    def __call__(self,sample):
        input, target = sample
        input += self.add_term
        return input, target 

if __name__ == "__main__": 

    dataset_normal = Tranform_Wine_Dataset()
    dataset_transform = Tranform_Wine_Dataset(transforms=Add_Tranform(term=100))
    
    #This works! The individual transforms: 
    # print(f'{dataset_normal[0]} \n {dataset_transform[0]}')

    ## Trying the compose feature to combine multiple tranforms: 

    compose = torchvision.transforms.Compose([Add_Tranform(term=100),Mul_Tranform()])
    dataset_compose = Tranform_Wine_Dataset(transforms=compose)

    print(f'Without Tranforms: {dataset_normal[10][0]} \n After Transforms: {dataset_compose[10][0]}')