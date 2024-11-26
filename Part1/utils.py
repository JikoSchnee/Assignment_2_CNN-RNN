# utils.py
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 1D tensor of predicted class indices.
        targets: 1D tensor of ground truth class indices.
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    return accuracy

def generate_data():
    """
    Generates synthetic training and test data using make_moons.
    Returns:
        train_loader: DataLoader for the training set.
        x_test: Tensor of test data features.
        y_test: Tensor of test data labels.
    """
    # Generate data using make_moons
    X, y = make_moons(n_samples=1000, noise=0.1)

    # Split into training and testing sets
    x_train = torch.tensor(X[:800], dtype=torch.float32)  # 800 samples for training
    y_train = torch.tensor(y[:800], dtype=torch.long)
    x_test = torch.tensor(X[800:], dtype=torch.float32)   # 200 samples for testing
    y_test = torch.tensor(y[800:], dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader, x_test, y_test

def plot_and_evaluate(model, train_losses, train_acc, val_losses, val_acc, testloader):
    """
    绘制训练和验证损失与准确率的图表，并评估模型在测试集上的表现。

    Parameters:
    - model: 训练好的模型
    - train_losses: 训练集损失
    - train_acc: 训练集准确率
    - val_losses: 验证集损失
    - val_acc: 验证集准确率
    - testloader: 测试数据加载器
    """
    # Plotting results
    plt.figure(figsize=(12, 5))

    # Plot Loss vs Epochs
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss vs Epochs")

    # Plot Accuracy vs Epochs
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy vs Epochs")

    plt.show()

    # Evaluate the model on the test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct_test / total_test:.2f}%")
