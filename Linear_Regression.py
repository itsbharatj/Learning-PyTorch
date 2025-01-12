## Implementing Linear Regression using NumpPy: 
import numpy as np

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

- Print the prediction after training
'''

## Initialize Varaibles: 

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([4, 6, 8, 10], dtype=np.float32)

w = 0.0 

## Loss Functions (Using MSE): 
def loss(y,y_pred): 
    l = (y_pred-y)**2
    return np.mean(l)

## Gradient Calculation: 
'''
Operations: 
1. Prediction of Y_pred: (x*w)
2. Calulating Loss: (Y-Y_pred)^2 ==> (Y-(x*w))^2 
3. Gradient Formula (dloss/dw)==> 2x * (Y-y_pred)
'''

# Gradient Calculation
def gradient(x, y, y_pred): 
    return np.mean(2*x*(y_pred-y))

# Training Loop
epoch = 1000
lr = 0.01  # Increased learning rate


for i in range(epoch): 
    prediction = X*w
    loss_val = loss(Y,prediction)
    grad = gradient(X,Y,prediction) ## Gradient will be an array --> Multiple Values. Do we need multiple values? 
    # print(f'Grad: {grad}')
    ## Update the weights: 

    '''
    w_new = w_old - lr*(dloss/dw)
    '''

    w = w - lr*grad
    if (i%2==0):
        print(f'Epoch: {i}, Loss: {loss_val:.3f} \n Prediction: {np.mean(prediction)}')

print(f'Final Prediction: {w*X} \n Weight: {w}')