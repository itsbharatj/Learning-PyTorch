import torch 

x = torch.tensor(4)
print(x)
w = torch.tensor(24.0, requires_grad=True)
print(w)

y = torch.tensor(15) ## This is the truth value, to which we want to compare our prediction against
## Forward Propogation to caculate the loss

y_h = x*w
loss =  y_h-y
# loss = 0.5*(z**2)

print(f'LOSS: {loss}')

## Calculating the rate of change of loss wrt to weights 

loss.backward() ## Shoud calculate dloss/dw

print(w.grad) ## This value is indicating that what is the change in the loss, if you change the wights by one unit

## Why should it be only scaler? Also, do need to understand 