import pandas as pd
import numpy as np 
from sklearn import datasets
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

'''
Steps: 

- Define and make a regression based dataset from Sklearn 
- Change the format to torch requirements (Row=Samples, Columns=Features)


- Define a linear model from torch
- Define the optimizer 
- Define the loss 

- Define the learing rate 
Training Loop: 
    - Make the prediction: from model.predict method
    - Calculate the loss, using the loss function 
    - Backpropogation: Calculate the gradients of loss wrt to weights 
    - Update the weights (optim.step()) method 
    - Set the gradients to zero by optim.zero_grad() function 
    - Print the prediciction for the model
'''

## 1) Defining the dataset from Sklearn: 

X,y = datasets.make_regression(n_samples=200, n_features=1)

# print(f'This is X: {X} \n This is y: {y}')
## Convert this to a torch tensor from a NumPy array: 

X_torch = torch.from_numpy(X).to(dtype=torch.float32)
y_torch = torch.from_numpy(y).to(dtype=torch.float32)

# print(type(X_torch), type(y_torch))

n_samples, n_features = X_torch.shape
n_inputs = n_features
n_outputs = 1
print(n_outputs) 

## 2) Defne the model, optimizer and loss: 

model = nn.Linear(in_features=n_inputs,out_features=n_outputs)
print(model.weight.dtype)

lr = 0.001
optim = torch.optim.SGD(model.parameters(),lr=lr)

loss = nn.MSELoss()

## 3) Define the training loop and train: 

n_epochs = 10000

for epoch in range(n_epochs):
    prediction = model(X_torch)
    l = loss(prediction, y_torch)

    l.backward()
    optim.step()

    optim.zero_grad()

    print(f'Epoch: {epoch}, Loss: {l}')


## 4) Plotting the final prediction: 


# plt.plot()
# prediction_y = model(X_torch).detach().numpy()

# # X = X.detach().numpy()
# plt.scatter(X,y,color='r')
# plt.plot(X,prediction_y,color='g')
# plt.show()


# Assuming model and X_torch are already defined
prediction_y = model(X_torch).detach().numpy()

# X should be converted to numpy if it's a tensor
X = X_torch.detach().numpy()

plt.scatter(X, y, color='r')
plt.plot(X, prediction_y, color='g')
plt.show()