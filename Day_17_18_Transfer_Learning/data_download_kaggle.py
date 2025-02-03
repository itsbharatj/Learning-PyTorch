import kagglehub
import shutil
import os

# Download latest version
dataset_handle_kaggle = "aleemaparakatta/cats-and-dogs-mini-dataset"
save_path = "data/"
path = kagglehub.dataset_download("aleemaparakatta/cats-and-dogs-mini-dataset")

print("Path to dataset files:", path)

shutil.move(path,save_path)
print("Data moved to data dir")
