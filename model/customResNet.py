import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Определим ResNet-блок, дальше из них будем строить глубокую ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Определим собственную версию ResNet
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        # Начальный слой (аналогично ResNet)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual блоки - у нас получится 6 блоков или эквивалент ResNet-6
        self.layer1 = self._make_layer(64, 2) # два блока по 64 каналов
        self.layer2 = self._make_layer(128, 2, stride=2) # переход к 128 каналам
        self.layer3 = self._make_layer(256, 2, stride=2) # переход к 256 каналам
        # Глобальный pooling и классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Преобразуем карту признаков в вектор размером batch_size x (256, 1, 1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )
            layers = []
            layers.append(ResNetBlock(self.in_channels, out_channels, stride,
            downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # [B, 64, H/2, W/2]
        x = self.maxpool(x) # [B, 64, H/4, W/4]
        x = self.layer1(x) # 64 → 64
        x = self.layer2(x) # 64 → 128
        x = self.layer3(x) # 128 → 256
        x = self.avgpool(x) # [B, 256, 1, 1]
        x = torch.flatten(x, 1) # [B, 256]
        x = torch.softmax(self.fc(x), dim=-1) # [B, num_classes]
        return x

# гипер параметры обучения
batch_size = 32
num_classes=3
num_epochs = 10
lr=0.001
# Загрузчики данных на основе наших датасетов
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

