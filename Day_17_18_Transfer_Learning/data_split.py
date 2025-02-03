from torch.utils.data import random_split
import os 

data_dir = '~/Desktop/pytorch_course/data/cats_dogs'
sets = ["train","val"]
folders = ["cats","dogs"]

for i in sets: 
    if not os.path.exists(f'{data_dir}/{i}'): 
        os.mkdir(f'{data_dir}/{i}') 
        os.mkdir(f'{data_dir}/{i}/cats')
        os.mkdir(f'{data_dir}/{i}/dogs')

files = []
for i in folders:
    files.append({i:os.listdir(f'{data_dir}/{i}')})
 
train_data = files["cats"]
val_data = 
