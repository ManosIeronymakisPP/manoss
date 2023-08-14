import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt

torch.manual_seed(42)  # Set random seed for reproducibility

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


# Define the model class
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # Input layer
        self.fc2 = nn.Linear(512, 256)  # Hidden layer
        self.fc3 = nn.Linear(256, 10)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the output layer
        x = self.fc3(x)
        return x

model = NeuralNet().to(device)  # Instantiate the model and move it to GPU if available

criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer

num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
total_correct = 0
total_samples = 0
plot_counter = 0  # Counter for the number of images plotted

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # Save a few sample images with predictions
        for i in range(images.shape[0]):
            if plot_counter >= 10:
                break  # Only save 10 images
            image = images[i].cpu().numpy()
            label = labels[i].cpu().item()
            prediction = predicted[i].cpu().item()

            # Reverse the normalization for saving the image
            image = (image * 0.3081) + 0.1307
            image = image.reshape(28, 28)  # Reshape the image to 28x28

            plt.imshow(image, cmap='gray')
            plt.title(f"True label: {label}, Predicted: {prediction}")
            plt.axis('off')
            plt.savefig(f"image_{plot_counter+1}.png")
            plt.close()
            plot_counter += 1

accuracy = 100 * total_correct / total_samples
print(f'Accuracy on the test set: {accuracy:.2f}%')
