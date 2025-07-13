import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from torchvision import transforms, datasets, models

CLASS_MAPPING = {
    'Potato_sick_late': 0,
    'Potato_sick_early': 1,
    'Potato_healthy': 2
}

def download_data(path_dataset):
    data = []
    y_train = []
    for path_dir in sorted(os.listdir(path=path_dataset)):
        path = path_dataset + path_dir + '/'
        for path_image in sorted(os.listdir(path=path)):
            image = Image.open(path + path_image).resize((180, 180))
            image = np.array(image)[:, :, :3]

            data.append(np.array(image.astype(np.uint8)))

            if 'Potato_sick_late' in path_dir:
                y_train.append(CLASS_MAPPING['Potato_sick_late'])
            elif 'Potato_sick_early' in path_dir:
                y_train.append(CLASS_MAPPING['Potato_sick_early'])
            elif 'Potato_healthy' in path_dir:
                y_train.append(CLASS_MAPPING['Potato_healthy'])

    return np.array(data), np.array(y_train)

# Так же определим класс датасета, подходящий для наших данных
class SimpleDataset():
  def __init__(self, data, y):
    self.data = data
    self.y = y
    self.transform = transforms.Compose(
        [transforms.ToTensor() ] # <-- это приводит HWC → CHW, т.е меняет местами порядок каналов в тензоре изображения
    )
    # стандартный порядок: ширина, высота, цветовые каналы
    # но пайторч требует: цветовые каналы, ширина, высота

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx]
    image = self.transform(image)
    target = self.y[idx]
    return image, torch.tensor(target).long()


