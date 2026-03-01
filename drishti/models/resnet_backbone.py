"""
Drishti-XAI-Defense
ResNet Backbone Module
"""

import torch.nn as nn
from torchvision import models


class DrishtiResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=True):
        super(DrishtiResNet, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)