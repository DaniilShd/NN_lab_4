import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from utilites.save import to_json
from data import download_data

CLASS_MAPPING = {
    'Potato_sick_late': 0,
    'Potato_sick_early': 1,
    'Potato_healthy': 2
}

# Определим ResNet-блок, дальше из них будем строить глубокую ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
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
        self.layer1 = self._make_layer(64, 2)   # два блока по 64 каналов
        self.layer2 = self._make_layer(128, 2, stride=2) # переход к 128 каналам
        self.layer3 = self._make_layer(256, 2, stride=2) # переход к 256 каналам

        # Глобальный pooling и классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Преобразуем карту признаков в вектор размером batch_size x (256, 1, 1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x = self.maxpool(x)                     # [B, 64, H/4, W/4]

        x = self.layer1(x)  # 64 → 64
        x = self.layer2(x)  # 64 → 128
        x = self.layer3(x)  # 128 → 256

        x = self.avgpool(x)             # [B, 256, 1, 1]
        x = torch.flatten(x, 1)         # [B, 256]
        x = torch.softmax(self.fc(x), dim=-1)   # [B, num_classes]
        return x

def train(num_epochs, num_classes, lr, train_loader, val_loader, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for epoch in range(num_epochs):
        # Цикл эпохи обучения:
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Train loss step: {loss.item()}")
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc.append(correct / total)
        train_loss.append(running_loss / len(train_loader))

        # Валидация
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc.append(val_correct / val_total)
        val_loss.append(val_running_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"Train acc: {train_acc[-1]:.3f} | Val acc: {val_acc[-1]:.3f}")

    to_json("metrics/custom_resnet/train", "train_accuracies", train_acc)
    to_json("metrics/custom_resnet/train", "train_losses", train_loss)
    to_json("metrics/custom_resnet/validation", "val_accuracies", val_acc)
    to_json("metrics/custom_resnet/validation", "val_losses", val_loss)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    plt.figure(figsize=(14, 4))
    for i in range(16):
        plt.subplot(2, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        title = ['sick_late', 'sick_early', 'healthy'][
                    preds[i]] + f"\n GT: {list(CLASS_MAPPING.keys())[labels[i].int().item()].replace('Potato_', '')}"
        plt.title(title)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img)
    plt.show()