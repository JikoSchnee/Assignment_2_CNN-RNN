import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)  # 32x32 RGB images flattened to 1D
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 output classes
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer (we'll use CrossEntropyLoss)
        return x


def train_model(model, trainloader, valloader, optimizer, criterion, epochs=10):
    model.train()
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training loss and accuracy
        train_losses.append(running_loss / len(trainloader))
        train_acc.append(100 * correct_train / total_train)

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(valloader))
        val_acc.append(100 * correct_val / total_val)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_acc[-1]:.2f}%")

        model.train()

    return train_losses, train_acc, val_losses, val_acc
