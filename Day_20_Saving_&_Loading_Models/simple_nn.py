'''
We need to save and load a model and then verify if they were the same or not: 

2 methods to save a model: 

- Just saving the model entirely (Also stores the instances and other file specific locations)
- Saving the state dictionary of the model (Preferred, as it only contains the weights and biases), need to define the model first though! 

Make a NN model 
- Save the model torch.save(model, PATH)

- load the model 
    - torch.load(PATH)

- Set the model to evaluation mode
'''

import torch 
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(self,input_size=5):
        super(Model,self).__init__()
        self.l1 = nn.Linear(input_size,4)
        self.l2 = nn.Linear(4,1)
    def forward(self,x): 
        out = self.l1(x)
        out = self.l2(out)
        return out
    
model = Model()


## Save the model: 
PATH = "models/simple_nn.pth"
# torch.save(model,PATH)

## Loading the model (Saving the entire model): 

model = torch.load(PATH)

for params in model.parameters(): 
    print(params)

model.eval()

for params in model.parameters(): 
    print(params)


## This works!! 

## Saving the model on the state dict
PATH_2 = f'models/simple_nn_dict.pth'
torch.save(model.state_dict(),PATH_2)

## Loading from the state dict: 

## Why do we need to define the model again? Will this not work, in case we do not do it? 

# l_model = Model()
# l_model.load_state_dict(torch.load(PATH_2)) 

# l_model.eval()


# for params in l_model.parameters(): 
#     print(params)


# l_model_2 = torch.load(PATH_2)



'Saving Checkpoint, using storing a dictionary:'

optimizer = torch.optim.Adam(model.parameters()) 

checkpoint = {
    "epoch":10,
    "model_state": model.state_dict(), 
    "optim_state": optimizer.state_dict()
}

## Save this optimizer and retrive the values later: 

# torch.save(checkpoint,"models/checkpoint_1.pth")

# l_checkpoint = torch.load("models/checkpoint_1.pth")

# epoch_number = l_checkpoint["epoch"]
# model_chk = Model()
# model_chk.load_state_dict(l_checkpoint['model_state'])
# optimizer.load_state_dict(l_checkpoint['optim_state'])

# for paramas in model_chk.parameters(): 
#     print(params)

# for paramas in optimizer.state_dict(): 
#     print(params)



## Loading and saving it from CPU and loading to MPS

## 1. CPU -> MPS 
'''
In this the model is saved as is, then moved to the MPS while loading
'''

model_cpu = Model()
PATH_3 = "models/model_cpu.pth"

# torch.save(model_cpu,PATH_3)

device = torch.device("mps")
model_cpu_mps = torch.load(PATH_3,map_location=device)

for params in model_cpu_mps.parameters(): 
    print(params)


'2. MPS --> CPU'

PATH_4 = "models/model_mps.pth"
torch.save(model_cpu_mps,PATH_3)

model_mps_cpu = torch.load(PATH, map_location=torch.device('cpu'))

for params in model_mps_cpu.parameters(): 
    print(params)
