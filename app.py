# app.py
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

# Define label names
labels = {0: "Arborio", 1: "Basmati", 2: "Ipsala", 3: "Jasmine", 4: "Karacadag"}

# Load your model class (from your CNN definition in Colab)
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, unique_classes=5):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128)
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(128 * 29 * 29, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, unique_classes)
        )
    def forward(self, X):
        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)
        return self.dense_layers(X)

# Instantiate and load model
model = CNN()
model.load_state_dict(torch.load("rice_classification_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create Flask app
