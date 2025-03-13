import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Define the model architecture
class SmartphoneCNN(nn.Module):
    def __init__(self):
        super(SmartphoneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjusted for 64x64 input
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Adjusted
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmartphoneCNN()
model.load_state_dict(torch.load("smartphone_model.pth", map_location=device))  # Updated path
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure input size matches model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess an image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to predict the class
def predict(image):
    image = preprocess_image(image)  # Preprocess the image
    with torch.no_grad():
        output = model(image)  # Get model predictions
        _, predicted = torch.max(output, 1)  # Get class with highest probability
    return "Fake" if predicted.item() == 0 else "Original"

# Streamlit UI
st.title("📱 Smartphone Authenticity Detector")
st.write("Upload an X-ray image to check if the smartphone is **original or fake**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        result = predict(image)
        st.write(f"### ✅ Prediction: {result}")
