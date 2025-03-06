## To check if GPU available
# import torch
# print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"GPU name: {torch.cuda.get_device_name(0)}")
# print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
# print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import transforms, datasets
import os
import shutil

# #######################################Organizing the dataset############################################
# Get the absolute path to the script's directory (where this file is located on local disk)
script_dir = os.path.dirname(os.path.abspath(__file__))

# # Path to the folder containing the images
# source_folder = os.path.join(script_dir, 'Braille Dataset')

# # Print the full path for debugging
# print(f"Looking for folder: {source_folder}")

# # Ensure the source folder exists
# if not os.path.exists(source_folder):
#     print(f"The folder {source_folder} does not exist.")
#     exit()

# print(f"Organizing files in: {source_folder}")

# # Iterate through all files in the source folder
# for filename in os.listdir(source_folder):
#     # Get the full path of the file
#     file_path = os.path.join(source_folder, filename)
    
#     # Skip directories, only process files
#     if os.path.isfile(file_path):
#         # Extract the letter from the filename (assuming the format is like "a1.JPG16whs")
#         letter = filename[0].lower()  # Get the first character and convert to lowercase
        
#         # Create a folder for the letter if it doesn't exist
#         letter_folder = os.path.join(source_folder, letter)
#         if not os.path.exists(letter_folder):
#             os.makedirs(letter_folder)
        
#         # Move the file to the corresponding folder
#         shutil.move(file_path, os.path.join(letter_folder, filename))

# print("Files have been organized successfully.")

# #######################################End of organizing the dataset############################################

#######################################Loading the dataset############################################

# Define transformations for grayscale images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Original size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # For grayscale images
])

# Set the path to your organized dataset
dataset_path = os.path.join(script_dir, 'Braille Dataset')

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Get class names
classes = dataset.classes
print(f"Classes: {classes}")
print(f"Number of images: {len(dataset)}")

# Get all indices
indices = list(range(len(dataset)))

# Shuffle indices
np.random.seed(1000)
np.random.shuffle(indices)

# Split indices into training, validation, and test sets
train_split = int(0.7 * len(indices))  # 70% training
val_split = int(0.85 * len(indices))  # 15% validation
train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

# Create samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

# Print the number of images in each set
print(f"Training images: {len(train_indices)}")
print(f"Validation images: {len(val_indices)}")
print(f"Test images: {len(test_indices)}")

# Verify a batch from the training loader
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}")
print(f"Labels: {labels}")

#######################################End of loading the dataset############################################