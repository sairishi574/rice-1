from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn

# Define label names
labels = {0: "Arborio", 1: "Basmati", 2: "Ipsala", 3: "Jasmine", 4: "Karacadag"}

app = Flask(__name__)

# Define CNN model
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

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.dense_layers(x)

# Instantiate and load model
model = CNN()
model.load_state_dict(torch.load("rice_classification_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3 RGB channels
])

@app.route('/')
def index():
    return "Rice Classification Model API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    # Accept key 'file' instead of 'image'
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': labels[predicted_class]})

if __name__ == "__main__":
    app.run(debug=True)


