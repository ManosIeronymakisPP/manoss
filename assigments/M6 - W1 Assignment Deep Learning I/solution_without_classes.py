import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)  # Set random seed for reproducibility
random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset using pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M6 - W1 Assignment Deep Learning I")

train_filepath = "mnist_train.csv"
test_filepath = "mnist_test.csv"

train_data = pd.read_csv(train_filepath)
test_data = pd.read_csv(test_filepath)

# Separate features and labels
train_labels = train_data['label']
train_features = train_data.drop('label', axis=1)

test_labels = test_data['label']
test_features = test_data.drop('label', axis=1)

# Convert the data to tensors
train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

# Define data loaders
batch_size = 128
num_workers = 2 if device.type == "cuda" else 0  # Adjust the number of workers based on GPU availability
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define the model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Randomly select 64 rows from X
random_indices = random.sample(range(len(test_features)), 64)
selected_images = test_features.iloc[random_indices].values

# Map each row back to a 28x28 grayscale image
selected_images = selected_images.reshape(-1, 28, 28)

# Display the images
fig, axs = plt.subplots(8, 8, figsize=(8, 8))
fig.suptitle('Randomly Selected Images')
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(selected_images[i*8+j], cmap='gray')
        axs[i, j].axis('off')
plt.tight_layout()
plt.show()
