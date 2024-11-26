from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train(FLAGS):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root=FLAGS.data_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True)

    testset = datasets.CIFAR10(root=FLAGS.data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

    # 初始化模型
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失

    # 用于记录训练过程中的损失和准确率
    loss_history = []
    accuracy_history = []

    print("Starting training loop...")

    # 训练循环
    for epoch in range(FLAGS.max_steps):
        print(f"Epoch {epoch + 1}/{FLAGS.max_steps}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for inputs, labels in trainloader:
            optimizer.zero_grad()  # 清除旧的梯度
            # print(f"Input shape: {inputs.shape}")  # 检查输入形状
            # print(f"Labels shape: {labels.shape}")  # 检查标签形状
            outputs = model(inputs)  # 正向传播
            # print(f"Output shape: {outputs.shape}")  # 检查输出形状
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # 计算训练准确率
            running_loss += loss.item()
            acc = accuracy(outputs, labels)
            running_accuracy += acc

        # 记录损失和准确率
        loss_history.append(running_loss / len(trainloader))
        accuracy_history.append(running_accuracy / len(trainloader))

        # 每个epoch结束后打印损失和准确率
        print(
            f'Epoch [{epoch + 1}/{FLAGS.max_steps}], Loss: {running_loss / len(trainloader)}, Accuracy: {running_accuracy / len(trainloader)}')

        # 每隔eval_freq步进行模型评估
        if (epoch + 1) % FLAGS.eval_freq == 0:
            evaluate(model, testloader)

    return loss_history, accuracy_history
    # # 在训练结束后绘制损失和准确率曲线
    # plot_metrics(loss_history, accuracy_history)

def evaluate(model, testloader):
    """
    Evaluate the model on the test set.
    """
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估时不需要计算梯度
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

def plot_metrics(loss_history, accuracy_history):
    """
    Plots the training loss and accuracy curves.
    """
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, label='Training Accuracy', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
