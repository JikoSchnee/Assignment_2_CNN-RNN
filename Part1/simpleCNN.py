import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import device
from torch.utils.data import DataLoader

# SimpleCNN Model Definition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Third convolution layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 10)  # Output layer for 10 classes (CIFAR-10)

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pooling
        x = x.view(-1, 128 * 4 * 4)  # Flatten the output to feed into fully connected layers
        x = torch.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.fc2(x)  # Output layer (logits)
        return x

# Training the model
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs):
    # Track losses and accuracies during training
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU or CPU
            optimizer.zero_grad()  # Zero gradients

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update model parameters

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation process
        val_loss, val_accuracy = evaluate_model(model, testloader, criterion)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    return train_losses, train_acc, val_losses, val_acc

# Evaluating the model
def evaluate_model(model, testloader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU or CPU
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return validation loss and accuracy
    val_loss = running_loss / len(testloader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy
