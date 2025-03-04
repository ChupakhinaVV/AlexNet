import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from dataset import CarsDataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

# Пути к файлам
data_path = "dataset/"
train_annos_path = os.path.join(data_path, "cars_train_annos.mat")
test_annos_path = os.path.join(data_path, "cars_test_annos_withlabels_eval.mat")
meta_path = os.path.join(data_path, "cars_meta.mat")

# Проверяем, что файлы существуют
if not os.path.exists(train_annos_path) or not os.path.exists(test_annos_path) or not os.path.exists(meta_path):
    print("Ошибка: не найдены файлы аннотаций! Проверь путь к файлам.")
    exit()

# Загружаем файлы
train_annos = scipy.io.loadmat(train_annos_path)
test_annos = scipy.io.loadmat(test_annos_path)
meta_data = scipy.io.loadmat(meta_path)

# Выводим ключи данных
print("Ключи в train_annos:", train_annos.keys())
print("Ключи в test_annos:", test_annos.keys())
print("Ключи в meta_data:", meta_data.keys())

# Проверяем структуру аннотаций
train_annotations = train_annos["annotations"]
test_annotations = test_annos["annotations"]

print("Тип данных train_annotations:", type(train_annotations))
print("Форма массива train_annotations:", train_annotations.shape)

# Выводим первую запись в аннотациях
print("Первая аннотация (train):", train_annotations[0][0])
# Извлекаем аннотации
train_annotations = train_annos["annotations"][0]  # (1, 8144) → берём первый массив
test_annotations = test_annos["annotations"][0]

# Исправляем метки (убираем двойное уменьшение!)
train_images = []
train_labels = []
train_bboxes = []

for ann in train_annotations:
    x1, y1, x2, y2, label, img_name = ann
    train_images.append(img_name[0])  # Имя файла
    train_labels.append(min(194, int(label[0][0]) - 1))
    train_bboxes.append((int(x1[0][0]), int(y1[0][0]), int(x2[0][0]), int(y2[0][0])))

# Проверяем первые 5 записей
for i in range(5):
    print(f"Изображение: {train_images[i]}, Класс: {train_labels[i]}, Рамка: {train_bboxes[i]}")

# Извлекаем список классов из meta_data
class_names = meta_data["class_names"][0]  # Это массив с названиями классов

# Создаём словарь {номер_класса: название}
class_dict = {i + 1: class_names[i][0] for i in range(len(class_names))}

# Проверяем первые 5 классов
for i in range(1, 6):
    print(f"Класс {i}: {class_dict[i]}")

# Проверяем классы для наших изображений
for i in range(5):
    print(f"Изображение: {train_images[i]}, Класс: {train_labels[i]} ({class_dict[train_labels[i]]})")
# Указываем путь к папке с изображениями (проверь правильность пути!)
# Проверяем, существует ли путь к изображению
image_folder = "dataset/cars_train/"  # Проверь, что папка существует
img_name = train_images[0]
img_path = os.path.join(image_folder, img_name)

if os.path.exists(img_path):
    print(f"Файл найден: {img_path}")
else:
    print(f"Файл НЕ найден! Проверь путь: {img_path}")

# Загружаем изображение
image = Image.open(img_path)
x1, y1, x2, y2 = train_bboxes[0]  # Берём координаты рамки

# Отображаем изображение
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
plt.title(f"Класс: {class_dict[train_labels[0]]}")  # Название машины
plt.axis("off")
plt.show()

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Проверяем исправленный диапазон классов
print(f"Исправленные классы: min={min(train_labels)}, max={max(train_labels)}")

# Создаём датасет
train_dataset = CarsDataset(train_images, train_labels, transform=transform)
print("Всего изображений в тренировочном датасете:", len(train_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Используем GPU, если доступно

# Загружаем предобученную AlexNet
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, len(class_dict))
model.to(device)

print(model)  # Выведет архитектуру

# Разбиваем данные на мини-пакеты (batch_size=32)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Проверим загрузку данных
images, labels = next(iter(train_loader))
print("Форма батча изображений:", images.shape)
print("Форма батча меток:", labels.shape)

criterion = nn.CrossEntropyLoss()  # Функция потерь (классификация)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Оптимизатор Adam

num_epochs = 7  # Количество эпох

for epoch in range(num_epochs):
    model.train()  # Переключаем в режим обучения
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(images)  # Прямой проход (forward)
        loss = criterion(outputs, labels)  # Вычисляем потери
        loss.backward()  # Обратное распространение (backpropagation)
        optimizer.step()  # Обновляем веса

        running_loss += loss.item()

    print(f"Эпоха [{epoch+1}/{num_epochs}], Потери: {running_loss/len(train_loader):.4f}")

print("Обучение завершено!")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность на обучающих данных: {100 * correct / total:.2f}%')

evaluate()

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

                # Вычисляем моменты градиента
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * grad ** 2

                # Корректируем моменты
                m_hat = state['m'] / (1 - beta1 ** state['step'])
                v_hat = state['v'] / (1 - beta2 ** state['step'])

                # Обновляем веса
                p.data -= group['lr'] * m_hat / (torch.sqrt(v_hat) + group['eps'])

adasmooth_optimizer = AdaSmooth(model.parameters(), lr=0.0001)

num_epochs = 7  # Количество эпох

for epoch in range(num_epochs):
    model.train()  # Переключаем в режим обучения
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        adasmooth_optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(images)  # Прямой проход (forward)
        loss = criterion(outputs, labels)  # Вычисляем потери
        loss.backward()  # Обратное распространение (backpropagation)
        adasmooth_optimizer.step()  # Обновляем веса

        running_loss += loss.item()

    print(f"Эпоха [{epoch+1}/{num_epochs}], Потери (AdaSmooth): {running_loss/len(train_loader):.4f}")

print("Обучение с AdaSmooth завершено!")

print("\nОценка модели, обученной с Adam:")
evaluate()

print("\nОценка модели, обученной с AdaSmooth:")
evaluate()