import kagglehub

# Download latest version
dataset_handle_kaggle = "aleemaparakatta/cats-and-dogs-mini-dataset"
save_path = "../data/"
path = kagglehub.dataset_download("aleemaparakatta/cats-and-dogs-mini-dataset")

print("Path to dataset files:", path)