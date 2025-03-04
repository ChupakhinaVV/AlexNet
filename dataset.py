import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CarsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # ✅ Не уменьшаем метки ещё раз!
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join("dataset/cars_train/", self.image_paths[idx])  # Путь к изображению
        image = Image.open(img_path).convert("RGB")  # Загружаем изображение
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
