from torch.utils.data import random_split
import os 
import shutil

'''
Data was downloaded in the form of: 
/data: 
    /class_1 
    /class_2 

Require: 
/data 
    /train
        /class_1
        /class_2
    /val
        /class_1 
        /class_2

Make the copies, copy the required data into the respective folders in the ratio
'''

data_dir = 'data/cats_dogs'
sets = ["train"]
folders = ["cats","dogs"]
cats_dir = f'{data_dir}/{folders[0]}'
dogs_dir = f'{data_dir}/{folders[1]}'

for i in sets: 
    if not os.path.exists(f'{data_dir}/{i}'): 
        os.mkdir(f'{data_dir}/{i}') 
        os.mkdir(f'{data_dir}/{i}/cats')
        os.mkdir(f'{data_dir}/{i}/dogs')

files = []
for i in folders:
    files.append({i:os.listdir(f'{data_dir}/{i}')})
 
ratio = 80
cats_train = int(len(os.listdir(cats_dir))*(ratio/100))
dogs_train = int(len(os.listdir(dogs_dir))*(ratio/100))

train_cats_files = os.listdir(cats_dir)[:cats_train]
train_dogs_files = os.listdir(dogs_dir)[:dogs_train]

val_cats_files = os.listdir(cats_dir)[cats_train:]
val_dogs_files = os.listdir(dogs_dir)[dogs_train:]

## Move the files: 

all_folders = [train_cats_files, train_dogs_files]
all_dest_folders = [f'{data_dir}/train/cats', f'{data_dir}/train/dogs']

for i, folder in enumerate(all_folders): 
    for file in folder: 
        shutil.move(f'{data_dir}/{folders[i]}/{file}', all_dest_folders[i])

if not os.path.exists(f'{data_dir}/val'): 
    os.mkdir(f'{data_dir}/val')
    for i in folders: 
        shutil.move(src=f'{data_dir}/{i}',dst=f'{data_dir}/val')