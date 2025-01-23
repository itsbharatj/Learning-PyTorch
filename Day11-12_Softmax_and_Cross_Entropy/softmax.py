import torch 
import numpy as np 

'''

Softmax: It is function used for classification tasks. Given the raw value of different classes, it 
normalizes and gives the probability of each of the class, given its value. 

Higher the relative value, higer is the probability of that class

The class with the highest probability can then be used as our prediction. 

The formula for softmax: 

f(x) == e^x/sum_of_all_classes(e^x)
'''

## Normal Implementation of Softmax: 

def softmax(ind,array): 
    sm = np.exp(array[ind])/np.sum(np.exp(array))
    return sm


classes = np.array([3.0, 2.0, 0.8],dtype=np.float32)
print(softmax(0,classes))

## Using Torch: 
## dim, they specify as to which axis to compute, 0: Rows

print(torch.softmax(torch.from_numpy(classes),dim=0))

# Using torch.nn.Softmax --> Only works after init 

tensor_classes = torch.from_numpy(classes)
softmax_layer = torch.nn.Softmax(dim=0)
softmax_output = softmax_layer(tensor_classes)
print(softmax_output)