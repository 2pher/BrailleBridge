## To check if GPU available
# import torch
# print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"GPU name: {torch.cuda.get_device_name(0)}")
# print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
# print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
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


#######################################Simple ANN Model############################################
class SimpleANN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=26, dropout_rate=0.5):
        super(SimpleANN, self).__init__()
        
        # Input layer (flatten 28x28 images to 784 vector)
        self.flatten = nn.Flatten()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Input layer
        x = self.flatten(x)  # Flatten the input (batch_size, 1, 28, 28) to (batch_size, 784)
        
        # First hidden layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        # Second hidden layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # Third hidden layer
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x

# # Create an instance of the model
# # Assuming you have 26 classes (a-z) for braille characters
# model = SimpleANN(input_size=28*28, num_classes=26)
# print(model)

#######################################End of Simple ANN Model############################################

#######################################Plotting the training curve############################################
def plot_training_curve(checkpoint_dir):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        checkpoint_dir: The directory containing the csv files produced during training
    """
    # Load the training and validation metrics
    train_loss = np.loadtxt(os.path.join(checkpoint_dir, "train_loss.csv"))
    val_loss = np.loadtxt(os.path.join(checkpoint_dir, "val_loss.csv"))
    train_acc = np.loadtxt(os.path.join(checkpoint_dir, "train_acc.csv"))
    val_acc = np.loadtxt(os.path.join(checkpoint_dir, "val_acc.csv"))

    # Plot training vs validation accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc)  # Number of epochs
    plt.plot(range(1, n+1), train_acc, label="Train")
    plt.plot(range(1, n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')

    # Plot training vs validation loss
    plt.subplot(1, 2, 2)
    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n+1), train_loss, label="Train")
    plt.plot(range(1, n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

#######################################End of plotting the training curve############################################

#######################################Training the model############################################
# Instantiate the model - make sure to use the correct number of classes
num_classes = len(dataset.classes)  # This should be 26 for a-z
model = SimpleANN(input_size=3*28*28, num_classes=num_classes)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Move model to device
model = model.to(device)

def train_net(net, train_loader, val_loader, batch_size=64, learning_rate=0.001, num_epochs=30, checkpoint_dir='checkpoints'):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Adam optimizer

    # Arrays to store training/validation metrics
    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the training data
        for inputs, labels in train_loader:
            # Move inputs and labels to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute training loss and accuracy
        train_loss[epoch] = running_loss / len(train_loader)
        train_acc[epoch] = 100 * correct / total

        # Validation step
        net.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                # Move inputs and labels to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Accumulate validation loss and accuracy
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Compute validation loss and accuracy
        val_loss[epoch] = val_running_loss / len(val_loader)
        val_acc[epoch] = 100 * val_correct / val_total

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss[epoch]:.4f}, Train Acc: {train_acc[epoch]:.2f}%, "
              f"Val Loss: {val_loss[epoch]:.4f}, Val Acc: {val_acc[epoch]:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss[epoch],
            'val_loss': val_loss[epoch],
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Print total training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")

    # Save training/validation metrics to CSV files
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt(os.path.join(checkpoint_dir, "train_loss.csv"), train_loss)
    np.savetxt(os.path.join(checkpoint_dir, "train_acc.csv"), train_acc)
    np.savetxt(os.path.join(checkpoint_dir, "val_loss.csv"), val_loss)
    np.savetxt(os.path.join(checkpoint_dir, "val_acc.csv"), val_acc)
    
    return train_loss, train_acc, val_loss, val_acc

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
checkpoint_dir = 'braille_ann_checkpoints'

# Train the model
train_loss, train_acc, val_loss, val_acc = train_net(
    net=model,
    train_loader=train_loader,
    val_loader=val_loader,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    checkpoint_dir=checkpoint_dir
)

# Plot the training curve
plot_training_curve(checkpoint_dir)
#######################################End of training the model############################################
