## Implementing Linear Regression using Optimizer Module of the PyTorch : 
import numpy as np
import torch

## You have a independent (X) and dependent (Y) with you --> Which has to be predicted, using the weights w
'''
Steps: 

- Model Prediction 
- Loss 
- Gradient Calculation

- Traing Loop 
    - Call eveything
    - Update the weights based on the gradient*learning rate (subtract that from the old weights)
    - Print the loss after every epoch, and the prediction that is given
    
    - Calculate the predicted value from w*X 
    - Then using the optimizer, calculate the gradients
    - From then use optim.step() --> What does it exactly does? --> Does the complete backpropogation (gradients calculation and weight update) 
    - Then it updates the weight automatically? 

- We can also: 
 = Use nn.Linear Layer for the forward propogation 
 = Not sure about the transformations and data types which PyTorch requires for the optimizer library

- Print the prediction after training
'''

## Initialize Varaibles: 

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
print(f'Val of w {w}')

## Loss Functions (Using MSE): 
def loss(y,y_pred): 
    l = (y_pred-y)**2
    return l.mean()

# loss = torch.nn.MSELoss()

## Gradient Calculation: 
'''
Operations: 
1. Prediction of Y_pred: (x*w)
2. Calulating Loss: (Y-Y_pred)^2 ==> (Y-(x*w))^2 
3. Gradient Formula (dloss/dw)==> 2x * (Y-y_pred)
'''

# Gradient Calculation
def gradient(x, y, y_pred): 
    return torch.mean(2*x*(y_pred-y))

# Training Loop
epoch = 100
lr = 0.01  # Increased learning rate
optim = torch.optim.SGD([w],lr)

for i in range(epoch): 
    prediction = X * w
    loss_val = loss(Y,prediction)
    # grad = gradient(X,Y,prediction) ## Gradient will be an array --> Multiple Values. Do we need multiple values? 
    
    optim.zero_grad()
    loss_val.backward()

    ## How the hell are the optim and the weights connected? 
    ## --> The loss_val is calculating the gradients wrt to weights, which is then getting used by the optim module for weight update
    optim.step()




    # with torch.no_grad(): 
    #     w -= lr*w.grad
    #     w.grad.zero_()

    # print(f'Grad: {grad}')
    ## Update the weights: 

    '''
    w_new = w_old - lr*(dloss/dw)
    '''

    # w = w - lr*grad
    if (i%2==0):
        print(f'Epoch: {i}, Loss: {loss_val:.3f} \n Prediction: {torch.mean(prediction)}')

print(f'Final Prediction for Torch: {w*X} \n Weight: {w}')