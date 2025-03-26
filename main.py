import os
import copy
import scipy.io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from collections import Counter

# region Загрузка аннотаций и мета-данных из .mat файлов
data_path = "dataset/"
train_annos_path = os.path.join(data_path, "cars_train_annos.mat")
test_annos_path = os.path.join(data_path, "cars_test_annos_withlabels_eval.mat")
meta_path = os.path.join(data_path, "cars_meta.mat")

if not (os.path.exists(train_annos_path) and os.path.exists(test_annos_path) and os.path.exists(meta_path)):
    print("Ошибка: не найдены файлы аннотаций! Проверь путь к файлам.")
    exit()

train_annos = scipy.io.loadmat(train_annos_path)
test_annos = scipy.io.loadmat(test_annos_path)
meta_data = scipy.io.loadmat(meta_path)

print("Ключи в train_annos:", train_annos.keys())
print("Ключи в test_annos:", test_annos.keys())
print("Ключи в meta_data:", meta_data.keys())
# endregion

# region Разбор аннотаций: извлекаем имена файлов, метки классов и bounding boxes
train_annotations = train_annos["annotations"][0]  # shape (1, 8144)
train_images = []
train_labels = []
train_bboxes = []

for ann in train_annotations:
    x1, y1, x2, y2, label, img_name = ann
    train_images.append(img_name[0])
    train_labels.append(int(label[0][0]) - 1)  # Приводим диапазон к 0-195
    train_bboxes.append((int(x1[0][0]), int(y1[0][0]), int(x2[0][0]), int(y2[0][0])))

# Выведем первые 5 аннотаций для проверки
for i in range(5):
    print(f"Изображение: {train_images[i]}, Класс: {train_labels[i]}, Рамка: {train_bboxes[i]}")
# endregion

# region Извлечение названий классов из meta-данных
class_names = meta_data["class_names"][0]
class_dict = {i + 1: class_names[i][0] for i in range(len(class_names))}
for i in range(1, 6):
    print(f"Класс {i}: {class_dict[i]}")
# endregion

# region Проверка существования изображений и визуализация одного примера
image_folder = "dataset/cars_train/"
img_path = os.path.join(image_folder, train_images[0])
if os.path.exists(img_path):
    print(f"Файл найден: {img_path}")
else:
    print(f"Файл не найден. Проверьте путь: {img_path}")

image = Image.open(img_path)
x1, y1, x2, y2 = train_bboxes[0]
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
plt.title(f"Класс: {class_dict[train_labels[0] + 1]}")
plt.axis("off")
plt.show()


# endregion

# region Определение Dataset
class CarsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join("dataset/cars_train/", self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# endregion

# region Трансформации и создание DataLoader
from sklearn.model_selection import train_test_split

print(f"Исправленные классы: min={min(train_labels)}, max={max(train_labels)}")
assert min(train_labels) >= 0, "Ошибка: есть отрицательные метки."
assert max(train_labels) <= 195, "Ошибка: метка выходит за пределы (195)."

# Разбиваем данные на train (80%) и val (20%)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_val = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CarsDataset(train_images, train_labels, transform=transform_train)
val_dataset = CarsDataset(val_images, val_labels, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Всего изображений в train: {len(train_dataset)}")
print(f"Всего изображений в val: {len(val_dataset)}")
# endregion


# # region Проверка данных и визуализация
# # Функция для обратной нормализации
# def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     img = img_tensor.clone().cpu().numpy().transpose((1, 2, 0))
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
#     return img
#
# # Получаем один батч из train_loader
# data_iter = iter(train_loader)
# images, labels = next(data_iter)
# print("Размер батча:", images.shape)
# print("Пример меток:", labels[:10].tolist())
#
# # Проверка распределения меток
# label_distribution = Counter(labels.tolist())
# print("Распределение меток в батче:", label_distribution)
#
# # Распределение по всему датасету:
# all_labels = []
# for _, lbls in train_loader:
#     all_labels.extend(lbls.tolist())
# print("Распределение меток в train_dataset:", Counter(all_labels))
#
# # Визуализация 5 изображений из батча
# for i in range(5):
#     img = denormalize(images[i])
#     plt.imshow(img)
#     plt.title(f"Метка: {labels[i].item()} (класс: {class_dict[labels[i].item() + 1]})")
#     plt.axis("off")
#     plt.show()
# # endregion

# region Создание моделей для Adam и AdaSmooth (обучение с нуля)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используемое устройство:", device)


# Объявляем класс AdaSmooth (должен быть объявлен до использования)
class AdaSmooth(optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(AdaSmooth, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                state['step'] += 1
                beta1, beta2 = group['beta1'], group['beta2']
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad ** 2
                m_hat = state['m'] / (1 - beta1 ** state['step'])
                v_hat = state['v'] / (1 - beta2 ** state['step'])
                p.data -= group['lr'] * m_hat / (torch.sqrt(v_hat) + group['eps'])


# Определяем модель AlexNet с нуля (с BatchNorm)
class AlexNetCustom(nn.Module):
    def __init__(self, num_classes=196):
        super(AlexNetCustom, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model_adam = AlexNetCustom(num_classes=196).to(device)
model_adam.apply(initialize_weights)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=3, verbose=True)

model_adasmooth = AlexNetCustom(num_classes=196).to(device)
model_adasmooth.apply(initialize_weights)
adasmooth_optimizer = AdaSmooth(model_adasmooth.parameters(), lr=0.001)

scheduler_adasmooth = torch.optim.lr_scheduler.ReduceLROnPlateau(
    adasmooth_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
# endregion


# region Определение функций обучения и оценки
def evaluate_model(model, device, val_loader, criterion):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(model, optimizer,scheduler, num_epochs, device, train_loader, val_loader, criterion):
    best_val_acc = 0
    patience, no_improve_epochs = 5, 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()



        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, val_acc = evaluate_model(model, device, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Эпоха [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")


        # Обновление learning rate через scheduler
        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc, no_improve_epochs = val_acc, 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Ранняя остановка: валидация не улучшается.")
                break

    print("Обучение завершено")
    return train_losses, val_losses, train_accuracies, val_accuracies


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# endregion

# region Запуск обучения и оценка моделей

num_epochs = 40

print("Обучение модели с нуля (Adam):")
train_losses_adam, val_losses_adam, train_accs_adam, val_accs_adam = train_model(
    model_adam, optimizer_adam, scheduler, num_epochs, device, train_loader, val_loader, criterion
)
_, accuracy_adam = evaluate_model(model_adam, device, val_loader, criterion)
print(f"Точность модели с Adam на TRAIN: {evaluate_model(model_adam, device, train_loader, criterion)[1]:.2f}%")
print(f"Валидационная точность модели с Adam: {accuracy_adam:.2f}%")

print("\nОбучение модели с нуля (AdaSmooth):")
train_losses_adasmooth, val_losses_adasmooth, train_accs_adasmooth, val_accs_adasmooth = train_model(
    model_adasmooth, adasmooth_optimizer, scheduler_adasmooth, num_epochs, device, train_loader, val_loader, criterion
)
_, accuracy_adasmooth = evaluate_model(model_adasmooth, device, val_loader, criterion)
print(
    f"Точность модели с AdaSmooth на TRAIN: {evaluate_model(model_adasmooth, device, train_loader, criterion)[1]:.2f}%")
print(f"Валидационная точность модели с AdaSmooth: {accuracy_adasmooth:.2f}%")

# endregion

# region Графики

plt.figure(figsize=(12, 5))
plt.plot(range(1, len(train_losses_adam) + 1), train_losses_adam, label="Train Loss (Adam)", marker="o")
plt.plot(range(1, len(val_losses_adam) + 1), val_losses_adam, label="Val Loss (Adam)", marker="o")
plt.plot(range(1, len(train_losses_adasmooth) + 1), train_losses_adasmooth, label="Train Loss (AdaSmooth)", marker="s")
plt.plot(range(1, len(val_losses_adasmooth) + 1), val_losses_adasmooth, label="Val Loss (AdaSmooth)", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss per Epoch")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(range(1, len(train_accs_adam) + 1), train_accs_adam, label="Train Acc (Adam)", marker="o")
plt.plot(range(1, len(val_accs_adam) + 1), val_accs_adam, label="Val Acc (Adam)", marker="o")
plt.plot(range(1, len(train_accs_adasmooth) + 1), train_accs_adasmooth, label="Train Acc (AdaSmooth)", marker="s")
plt.plot(range(1, len(val_accs_adasmooth) + 1), val_accs_adasmooth, label="Val Acc (AdaSmooth)", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Train & Validation Accuracy per Epoch")
plt.show()

# endregion
