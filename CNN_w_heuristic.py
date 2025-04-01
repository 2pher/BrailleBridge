####################################################################################
# Brief Description:
# This script loads a pre-trained CNN model (braille_cnn_model.pth) for character classification and uses it to predict
# the characters in segmented braille images. The predictions are then concatenated into a string,
# which is further processed by a heuristic function to segment the string into words or sentences.

# Note:
# The images are loaded from the segmented_braille_images directory

####################################################################################

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

from character_classifier_CNN import CNN
from sentence_heuristic import segment_sentence


if __name__ == "__main__":
    # load the model
    model = CNN()
    model.load_state_dict(torch.load('braille_cnn_model.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For 3 channels
    ])

    # Alphabet mapping
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Get all image files from the directory
    image_dir = 'Sequence'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process all images and collect predictions
    predictions = []
    for image_file in image_files:
        try:
            # Load and transform image
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            # Map to letter
            predicted_letter = alphabet[prediction]
            predictions.append(predicted_letter)
            
            print(f"Processed {image_file}: Predicted '{predicted_letter}'")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    final_output = ''.join(predictions)
    print(f"\nFinal concatenated prediction: {final_output}")

    print(f"Segmented output: {segment_sentence(final_output)}")