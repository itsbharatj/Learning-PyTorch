import torch
import numpy as np
import torch.nn as nn 
'''
Cross Entropy is a loss metric that is used for multiclass classification tasks 
(Binary Cross Entropy for predictions for one class)

This compares the actual prediction with the probabilty of that prediction. 

If the prediction of the correct class is high then it gives a low loss 
If the prediction of the wrong class is higher (and correct class is lower) then a high loss is given 

The softmax: Takes the raw values of the classes and gives a normalized value of them
Cross Entropy Loss: Takes on these probabilites and then computes a loss for the weight update 

NOTE: In the PyTorch implementation - the function requies the raw values, as it automatically applies the softmax operation for the same!  
'''

loss = nn.CrossEntropyLoss() ##This is the init 
y = torch.tensor([1]) ## This indicates for the first sample, the 2nd class is the actual class

y_pred_good = torch.tensor([[100, 8.9,0.1]],dtype=torch.float32)
y_pred_bad = torch.tensor([[15.9,8.0,0.1]],dtype=torch.float32)

loss_good = loss(y_pred_good,y)
loss_bad = loss(y_pred_bad,y)

print(f'Loss Good: {loss_good} \nLoss Bad: {loss_bad}')



