import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import io
import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorflow import keras
from data import download_data, SimpleDataset
import tensorflow as tf
from resnet18 import train_resnet18
from utilites.show_image import show_all_plants
from simple_resnet import train

base_path = "data"
# гипер параметры обучения
batch_size = 32
num_classes=3
num_epochs = 10
lr=0.001

if __name__ == "__main__":

    train_dataset, y_train = download_data(f"{base_path}/Train/")
    train_dataset = SimpleDataset(train_dataset, y_train)

    test_dataset, y_test = download_data(f"{base_path}/Test/")
    test_dataset = SimpleDataset(test_dataset, y_test)

    # Загрузчики данных на основе наших датасетов
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train(num_epochs, num_classes, lr, train_loader, val_loader, test_dataset)


    train_resnet18(num_epochs, num_classes, lr, train_loader, val_loader, test_dataset)







