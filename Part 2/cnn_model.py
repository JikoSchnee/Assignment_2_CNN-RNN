from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, n_channels = 3, n_classes = 10):
    """
    Initializes CNN object.

    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
    self.fc1 = nn.Linear(512 * 2 * 2, 1024)  # 512 channels, image size 32x32 -> 8x8 after pooling
    self.fc2 = nn.Linear(1024, n_classes)  # CIFAR-10 has 10 classes

  def forward(self, x):
    """
    Performs forward pass of the input.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))

    # Calculate the flattened size dynamically based on the image size
    x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer

    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
