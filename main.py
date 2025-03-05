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

#region Загрузка аннотаций и мета-данных из .mat файлов

# Пути к файлам
data_path = "dataset/"
train_annos_path = os.path.join(data_path, "cars_train_annos.mat")
test_annos_path = os.path.join(data_path, "cars_test_annos_withlabels_eval.mat")
meta_path = os.path.join(data_path, "cars_meta.mat")

if not (os.path.exists(train_annos_path) and os.path.exists(test_annos_path) and os.path.exists(meta_path)):
    print("Ошибка: не найдены файлы аннотаций! Проверь путь к файлам.")
    exit()

# Загружаем файлы
train_annos = scipy.io.loadmat(train_annos_path)
test_annos = scipy.io.loadmat(test_annos_path)
meta_data = scipy.io.loadmat(meta_path)

print("Ключи в train_annos:", train_annos.keys())
print("Ключи в test_annos:", test_annos.keys())
print("Ключи в meta_data:", meta_data.keys())

#endregion

#region Разбор аннотаций: извлекаем имена файлов, метки классов и bounding boxes

# Извлекаем массив аннотаций (train)
train_annotations = train_annos["annotations"][0]  # shape (1, 8144) → берем первый элемент

train_images = []
train_labels = []
train_bboxes = []

# Для каждой аннотации распаковываем данные
for ann in train_annotations:
    x1, y1, x2, y2, label, img_name = ann
    train_images.append(img_name[0])  # Имя файла
    # Если оригинальные метки идут от 1 до 196, то вычитаем 1, при этом ограничиваем максимальное значение до 194.
    train_labels.append(min(194, int(label[0][0]) - 1))
    train_bboxes.append((int(x1[0][0]), int(y1[0][0]), int(x2[0][0]), int(y2[0][0])))

# Выведем первые 5 аннотаций для проверки
for i in range(5):
    print(f"Изображение: {train_images[i]}, Класс: {train_labels[i]}, Рамка: {train_bboxes[i]}")

#endregion

#region Извлечение названий классов из meta-данных

class_names = meta_data["class_names"][0]  # Массив названий классов
# Создаем словарь: ключ = номер класса (от 1 до 196), значение = название класса
class_dict = {i + 1: class_names[i][0] for i in range(len(class_names))}

# Для проверки выведем первые 5 классов:
for i in range(1, 6):
    print(f"Класс {i}: {class_dict[i]}")

#endregion

#region Проверка существования изображений и визуализация одного примера

image_folder = "dataset/cars_train/"  # Папка с изображениями
img_path = os.path.join(image_folder, train_images[0])
if os.path.exists(img_path):
    print(f"Файл найден: {img_path}")
else:
    print(f"Файл не найден. Проверьте путь: {img_path}")

# Загружаем и отображаем изображение с bounding box
image = Image.open(img_path)
x1, y1, x2, y2 = train_bboxes[0]
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
plt.title(f"Класс: {class_dict[train_labels[0] + 1]}")
plt.axis("off")
plt.show()

#endregion

#region Определение Dataset

class CarsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # Метки уже обработаны в main.py, не уменьшаем здесь!
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

#endregion

#region Трансформации и создание DataLoader

transform = transforms.Compose([
    transforms.RandomRotation(15),  # Вращение на ±15 градусов
    transforms.RandomHorizontalFlip(),  # Горизонтальное отражение
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение цвета
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Финальная проверка диапазона меток
print(f"Исправленные классы: min={min(train_labels)}, max={max(train_labels)}")
assert min(train_labels) >= 0, " Ошибка: есть отрицательные метки"
assert max(train_labels) <= 194, " Ошибка: метка выходит за пределы (194)"

train_dataset = CarsDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("Всего изображений в тренировочном датасете:", len(train_dataset))

#endregion

#region Создание валидационного набора

from sklearn.model_selection import train_test_split

# Разделяем данные на train (80%) и val (20%)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Создаем датасеты
train_dataset = CarsDataset(train_images, train_labels, transform=transform)
val_dataset = CarsDataset(val_images, val_labels, transform=transform)

# Создаем DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Валидацию не мешаем

# Проверяем количество изображений
print(f"Всего изображений в train: {len(train_dataset)}")
print(f"Всего изображений в val: {len(val_dataset)}")

#endregion

#region Создание независимых моделей для Adam и AdaSmooth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаем исходную модель с предобученными весами и нужным выходным слоем
initial_model = models.alexnet(pretrained=True)
initial_model.classifier[6] = nn.Linear(4096, len(class_dict))  # Здесь len(class_dict) = 196
initial_model.to(device)
# Сохраняем начальное состояние модели
initial_state = copy.deepcopy(initial_model.state_dict())

# Модель для Adam
model_adam = models.alexnet(pretrained=False)
model_adam.classifier[6] = nn.Linear(4096, len(class_dict))
model_adam.load_state_dict(initial_state)
model_adam.to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.00005, weight_decay=1e-4)

# Модель для AdaSmooth
model_adasmooth = models.alexnet(pretrained=False)
model_adasmooth.classifier[6] = nn.Linear(4096, len(class_dict))
model_adasmooth.load_state_dict(initial_state)
model_adasmooth.to(device)

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
adasmooth_optimizer = AdaSmooth(model_adasmooth.parameters(), lr=0.0001)

#endregion

#region Определение функций обучения и оценки

def train_model(model, optimizer, num_epochs, device, train_loader, val_loader, criterion):
    best_val_acc = 0
    patience = 2
    no_improve_epochs = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)

        # Валидация
        avg_val_loss, val_acc = evaluate_model(model, device, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Эпоха [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Ранняя остановка
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0  # Сброс счетчика
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Ранняя остановка: валидация не улучшается.")
                break  # Прерываем обучение

    print("Обучение завершено")
    return train_losses, val_losses, train_accuracies, val_accuracies



# Оценка модели
def evaluate_model(model, device, val_loader, criterion):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
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

criterion = nn.CrossEntropyLoss()

#endregion

#region Обучение и оценка моделей

num_epochs = 20

print("Обучение модели с Adam:")
train_losses_adam, val_losses_adam, train_accs_adam, val_accs_adam = train_model(
    model_adam, optimizer_adam, num_epochs, device, train_loader, val_loader, criterion
)

# Оцениваем точность модели на TRAIN и VALIDATION наборах
accuracy_adam_train = evaluate_model(model_adam, device, train_loader, criterion)[1]  # Train Accuracy
accuracy_adam_val = evaluate_model(model_adam, device, val_loader, criterion)[1]  # Validation Accuracy

print(f"Точность модели с Adam на тренировочном наборе: {accuracy_adam_train:.2f}%")
print(f"Валидационная точность модели с Adam: {accuracy_adam_val:.2f}%")

print("\nОбучение модели с AdaSmooth:")
train_losses_adasmooth, val_losses_adasmooth, train_accs_adasmooth, val_accs_adasmooth = train_model(
    model_adasmooth, adasmooth_optimizer, num_epochs, device, train_loader, val_loader, criterion
)

# Оцениваем точность модели на TRAIN и VALIDATION наборах
accuracy_adasmooth_train = evaluate_model(model_adasmooth, device, train_loader, criterion)[1]  # Train Accuracy
accuracy_adasmooth_val = evaluate_model(model_adasmooth, device, val_loader, criterion)[1]  # Validation Accuracy

print(f"Точность модели с AdaSmooth на тренировочном наборе: {accuracy_adasmooth_train:.2f}%")
print(f"Валидационная точность модели с AdaSmooth: {accuracy_adasmooth_val:.2f}%")

#endregion

#region Графики

# График потерь
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

# График точности
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(train_accs_adam) + 1), train_accs_adam, label="Train Acc (Adam)", marker="o")
plt.plot(range(1, len(val_accs_adam) + 1), val_accs_adam, label="Val Acc (Adam)", marker="o")
plt.plot(range(1, len(train_accs_adasmooth) + 1), train_accs_adasmooth, label="Train Acc (AdaSmooth)", marker="s")
plt.plot(range(1, len(val_accs_adasmooth) + 1), val_accs_adasmooth, label="Val Acc (AdaSmooth)", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Validation Accuracy per Epoch")
plt.show()

#endregion
