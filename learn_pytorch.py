import torch 

weights = torch.ones(4,requires_grad=True)

# for i in range(5): 
#     output = (weights*3).sum()
#     output.backward()
#     print(weights.grad)


## Optimziers: 

# for i in range(5): 
optimizer = torch.optim.SGD(weights,lr=0.01)
optimizer.step()
optimizer.zero_grad()
    # print(weights.grad)

