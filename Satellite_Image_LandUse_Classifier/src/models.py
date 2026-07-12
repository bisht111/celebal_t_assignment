import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights


class BaselineCNN(nn.Module):
    """
    Simple 3-layer CNN used as the baseline model.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
   


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 Transfer Learning Model
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Load pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    # --------------------------
    # Phase 1
    # Freeze backbone
    # --------------------------
    def freeze_backbone(self):

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    # --------------------------
    # Phase 2
    # Unfreeze last two blocks
    # --------------------------
    def unfreeze_last_blocks(self):

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

    # --------------------------
    # Embedding extractor
    # (Module 2)
    # --------------------------
    def extract_features(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)

        x = torch.flatten(x, 1)

        return x