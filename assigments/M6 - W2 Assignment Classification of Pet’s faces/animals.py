import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Set the paths
image_dir = r'C:\Users\ManosIeronymakisProb\OneDrive - Probability\Bureaublad\ELU\M6 - W2 Assignment Classification of Pet’s faces\images'
names_file = r'C:\Users\ManosIeronymakisProb\OneDrive - Probability\Bureaublad\ELU\M6 - W2 Assignment Classification of Pet’s faces\trainval.txt'

# Step 1: Visualize at least 20 images on a grid

# Load the names from the file
with open(names_file, 'r') as file:
    names = file.read().splitlines()

# Create a figure and axes for the grid
fig, axs = plt.subplots(5, 4, figsize=(12, 15))

# Loop through the images and plot them
for i, ax in enumerate(axs.flat):
    image_path = os.path.join(image_dir, names[i] + '.jpg')
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.set_title(names[i])
    ax.axis('off')

# Display the grid of images
plt.tight_layout()
plt.show()

# Step 2: Prepare the dataset for Deep Learning

# Define the data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
dataset = ImageFolder(image_dir, transform=transform)

# Step 3: Do a train-test split (preserve label variation)

# Split the dataset into train and test sets using stratified shuffle split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(dataset.samples, dataset.targets))

# Create train and test datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Step 4: Train the Convolutional Neural Network and plot accuracy

# Define the number of classes
num_classes = len(dataset.classes)

# Create data loaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)  # Adjust the number of output neurons based on the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create an instance of the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_accuracy = 100.0 * correct / total
    train_loss = running_loss / len(train_loader)

    # Evaluation on the test set
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total

    # Append accuracy and loss to lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Print accuracy and loss for each epoch
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Plot the accuracy
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# Step 5: Optional - Create a separate model for cats and dogs classification

# Adjust the labels for cats and dogs classification
cat_dog_dataset = ImageFolder(image_dir, transform=transform)
cat_dog_dataset.class_to_idx = {'cat': 0, 'dog': 1}

# Split the adjusted dataset into train and test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
cat_dog_train_indices, cat_dog_test_indices = next(splitter.split(cat_dog_dataset.samples, cat_dog_dataset.targets))

# Create train and test datasets for cats and dogs classification
cat_dog_train_dataset = torch.utils.data.Subset(cat_dog_dataset, cat_dog_train_indices)
cat_dog_test_dataset = torch.utils.data.Subset(cat_dog_dataset, cat_dog_test_indices)

# We repeat steps 4 and 5 with the adjusted datasets for cats and dogs classification

# Step 6: Building a new CNN

# Create data loaders for train and test datasets for cats and dogs classification
cat_dog_train_loader = DataLoader(cat_dog_train_dataset, batch_size=32, shuffle=True)
cat_dog_test_loader = DataLoader(cat_dog_test_dataset, batch_size=32, shuffle=False)

# Create an instance of the CNN model for cats and dogs classification
cat_dog_model = CNN()

# Define the loss function and optimizer for cats and dogs classification
cat_dog_criterion = nn.CrossEntropyLoss()
cat_dog_optimizer = optim.SGD(cat_dog_model.parameters(), lr=0.001, momentum=0.9)

# Training loop for cats and dogs classification
cat_dog_num_epochs = 10
cat_dog_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cat_dog_model.to(cat_dog_device)

cat_dog_train_losses = []
cat_dog_train_accuracies = []
cat_dog_test_accuracies = []

for epoch in range(cat_dog_num_epochs):
    # Set model to training mode
    cat_dog_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in cat_dog_train_loader:
        images = images.to(cat_dog_device)
        labels = labels.to(cat_dog_device)

        cat_dog_optimizer.zero_grad()
        outputs = cat_dog_model(images)
        loss = cat_dog_criterion(outputs, labels)
        loss.backward()
        cat_dog_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_accuracy = 100.0 * correct / total
    train_loss = running_loss / len(cat_dog_train_loader)

    # Evaluation on the test set for cats and dogs classification
    cat_dog_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in cat_dog_test_loader:
            images = images.to(cat_dog_device)
            labels = labels.to(cat_dog_device)

            outputs = cat_dog_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total

    # Append accuracy and loss to lists for cats and dogs classification
    cat_dog_train_losses.append(train_loss)
    cat_dog_train_accuracies.append(train_accuracy)
    cat_dog_test_accuracies.append(test_accuracy)

    # Print accuracy and loss for each epoch for cats and dogs classification
    print(f'Epoch {epoch+1}/{cat_dog_num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Plot the accuracy for cats and dogs classification
plt.plot(range(1, cat_dog_num_epochs + 1), cat_dog_train_accuracies, label='Train')
plt.plot(range(1, cat_dog_num_epochs + 1), cat_dog_test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Define the CNN model with the pyramid architecture, activation functions (ReLU), and max pooling
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 28 * 28, 2)  # Adjust the number of output neurons to 2 for cats and dogs classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create an instance of the CustomCNN model
custom_model = CustomCNN()

# Define the loss function and optimizer for the CustomCNN model
custom_criterion = nn.CrossEntropyLoss()
custom_optimizer = optim.SGD(custom_model.parameters(), lr=0.001, momentum=0.9)

# Training loop for the CustomCNN model
custom_num_epochs = 10
custom_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custom_model.to(custom_device)

custom_train_losses = []
custom_train_accuracies = []
custom_test_accuracies = []

for epoch in range(custom_num_epochs):
    # Set model to training mode
    custom_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(custom_device)
        labels = labels.to(custom_device)

        custom_optimizer.zero_grad()
        outputs = custom_model(images)
        loss = custom_criterion(outputs, labels)
        loss.backward()
        custom_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_accuracy = 100.0 * correct / total
    train_loss = running_loss / len(train_loader)

    # Evaluation on the test set for the CustomCNN model
    custom_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(custom_device)
            labels = labels.to(custom_device)

            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total

    # Append accuracy and loss to lists for the CustomCNN model
    custom_train_losses.append(train_loss)
    custom_train_accuracies.append(train_accuracy)
    custom_test_accuracies.append(test_accuracy)

    # Print accuracy and loss for each epoch for the CustomCNN model
    print(f'Epoch {epoch+1}/{custom_num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Plot the accuracy for the CustomCNN model
plt.plot(range(1, custom_num_epochs + 1), custom_train_accuracies, label='Train')
plt.plot(range(1, custom_num_epochs + 1), custom_test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


