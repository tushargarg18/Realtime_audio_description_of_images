# encoder.py

import torch
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, embed_size=256, pretrained=True):
        super(CNNEncoder, self).__init__()

        #resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        modules = list(resnet.children())[:-1]  # Remove FC layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten to [B, 512]
        features = self.fc(features)                    # Project to [B, embed_size]
        features = self.bn(features)
        features = self.dropout(features)
        return features