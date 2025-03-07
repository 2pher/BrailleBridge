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
import math

from PIL import Image
from torchvision import transforms, datasets
import os
import shutil

from main import *

#######################################CNN LeNet Model############################################

class CNN(nn.Module):

    def __init__(self, kernel_sizes=[5, 5], conv_dim=[64,128],hidden_sizes=[120, 84], num_classes=26, dropout_rate=0.25, input_dim=28):
        super(CNN, self).__init__()

        #define convolutional layers and pooling layers
        self.conv1 = nn.Conv2d(3, conv_dim[0], kernel_sizes[0])
        self.conv2 = nn.Conv2d(conv_dim[0], conv_dim[1], kernel_sizes[1])
        self.pool = nn.MaxPool2d(2, 2)

        #calculate output size into linear layers
        self.size = math.floor((input_dim - kernel_sizes[0] + 1) / 2)
        self.size = math.floor((self.size - kernel_sizes[1] + 1) / 2)

        self.conv_output = conv_dim[1]

        #define linear layers and dropout
        self.fc1 = nn.Linear(conv_dim[1] * self.size * self.size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        #convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #flatten into linear layers
        x = x.view(-1, self.conv_output * self.size * self.size)

        #linear layers and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

#######################################End of CNN LeNet Model############################################

#######################################Training the model############################################
# Instantiate the model - make sure to use the correct number of classes
num_classes = len(dataset.classes)  # This should be 26 for a-z
model = CNN()

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Define hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 30
checkpoint_dir = 'braille_cnn_checkpoints'

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

#######################################Evaluate the model###################################################

err, loss = evaluate(
            net=model,
            loader=test_loader,
            criterion=nn.CrossEntropyLoss()
            )

print('Test Error: ', err)
print('Test Loss: ', loss)
print('Test accuracy: ', 1-err)

#######################################End of Evaluate the model###################################################
