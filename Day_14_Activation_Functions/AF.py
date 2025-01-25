'''
So we need to implement multiple activation functions. 

These are some of the main AFs and their use case: 

1. Step Function: 
    - If the value of x is greater than a certain threshold, then it returns 1 ow 0 
    - Not really used anywhere 

2. Sigmoid Activation Function: 
    - This gives a value between 0 and 1. At x=0 the value is 0.5. Slowly it becomes 1 as x increases 
    - Used for Binary Classification Tasks
    - Used at the last layer of a Binary Classification Task (Given one raw value, give it a yes or no)

3. TanH: 
    - Gives between 0 - 1 
    - It is centred at 0 
    - Used for hidden layers

4. ReLU: 
    - Most popular activation function there is
    - if the value of x > 0 then y = x, else y = 0 is x < 0 
    - The default activation function used for the hidden layers 
    - It might be susceptible to the vanishing gradients problem: 
        During backpropogation, the derivative of change of weights/loss becomes very small as, the value change is little? And then multiplied by a small learning rate - it becomes even more small!

5. Leaky ReLU: 
    - This is a slightly modified version of ReLU where if x < 0 then y is not equal to 0, but it is a very small number
    - y = x, if x > 0 
    - Solves the vanishing gradients problem

Some Questions: 
    - How does the derivative of the weights and loss becomes small after applying ReLU 
    - How exactly is leaky ReLU able to solve this problem? By getting the values not exactly 0 
    - Also I need to understand what is the need for these functions to close the ones with the values which are less than 0, 
    do the raw values out of a NN signify anything? If the value is 0, then it considered closed.

6. Softmax: 
    - This normalizes the raw values from different classes to it's corresponding probabilities
    - It is used for multiclass classifications


USAGE: 

Can use activation functions in two ways: 

1. During initilization (torch.nn.a_f)

    Eg. torch.nn.sigmoid()

2. At the forward method (torch.A_F)

    Eg. torch.Sigmoid()

- Why are there two ways 
- Why does the nn methods have to be initialized (or do they) to be used  

Will make a simple NN, with 2 liner layers and one activation funtions. Both the ways 
'''

