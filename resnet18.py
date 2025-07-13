from utilites.save import to_json
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

CLASS_MAPPING = {
    'Potato_sick_late': 0,
    'Potato_sick_early': 1,
    'Potato_healthy': 2
}

def train_resnet18(num_epochs, num_classes, lr, train_loader, val_loader, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Загрузим предобученную модель ResNet-18
    model = models.resnet18(pretrained=True)

    # Заменим последний полносвязный слой (fc) под нашу задачу
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Softmax(dim=-1)
    )

    # Переносим на GPU
    model = model.to(device)

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
                _, predicted = torch.max(outputs, dim=1)  # предсказанные классы
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc.append(val_correct / val_total)
        val_loss.append(val_running_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"Train acc: {train_acc[-1]:.3f} | Val acc: {val_acc[-1]:.3f}")

    to_json("metrics/resnet18/train", "train_accuracies", train_acc)
    to_json("metrics/resnet18/train", "train_losses", train_loss)
    to_json("metrics/resnet18/validation", "val_accuracies", val_acc)
    to_json("metrics/resnet18/validation", "val_losses", val_loss)

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