import torch
import torch.nn as nn
import torch.nn.functional as F


# Class for encoder model
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)   # output: [B, 64, 224, 224]
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for conv1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # output: [B, 128, 112, 112]
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for dimensionality reduction
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # output: [B, 512]
        self.fc2 = nn.Linear(512, 256)           # final feature vector: [B, 256]

        self.relu = nn.ReLU()

    def forward(self, x):
        # Feature extraction
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Dimensionality reduction
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x  # [B, 256]