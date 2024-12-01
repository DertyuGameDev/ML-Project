import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'


class HumanPoseDataset(Dataset):
    """Класс для хранения меток и картинок из датасета"""
    def __init__(self, img_dir, labels_df=None, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.img_files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.labels_df) if self.labels_df is not None else len(os.listdir(self.img_dir))
    
    def __getitem__(self, index):
        img_name = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels_df is not None:
            img_id = int(os.path.splitext(img_name)[0])
            label = self.labels_df.loc[self.labels_df["img_id"] == img_id, "target_feature"].values[0]
            return image, label
        else:
            return image, img_name
        

def show_images(dataset, num_images=5):
    """Функция для отображения нескольких изображений из датасета"""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]  # Получаем изображение и метку
        image = image.permute(1, 2, 0).numpy()  # Меняем порядок осей для отображения
        image = (image * 0.5 + 0.5)  # Обратно нормализуем для отображения (если нормализовали)
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    plt.show()


def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr): 
    """Функция для отображения нескольких изображений из датасета"""
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    train_losses, val_losses = [], [] 
    val_metrics = [] 

    for epoch in range(num_epochs): 
        # Обучение 
        model.train() 
        train_loss = 0 
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device) 

            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            train_loss += loss.item() 

        train_loss /= len(train_loader) 
        train_losses.append(train_loss) 
        
        # Оценка 
        model.eval() 
        val_loss = 0
        all_preds = [] 
        all_labels = [] 
        with torch.no_grad(): 
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device) 
                outputs = model(images)
                loss = criterion(outputs, labels) 
                val_loss += loss.item()
            
                preds = torch.argmax(outputs, dim=1) 
                all_preds.extend(preds.cpu().numpy()) 
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss) 
        
        acc = accuracy_score(all_labels, all_preds) 
        prec = precision_score(all_labels, all_preds, average="weighted") 
        rec = recall_score(all_labels, all_preds, average="weighted") 
        f1 = f1_score(all_labels, all_preds, average="weighted") 
        val_metrics.append((acc, prec, rec, f1)) 
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Val Prec: {prec:.4f}, Val Recall: {rec:.4f}, Val F1: {f1:.4f}") 
        
    return train_losses, val_losses, val_metrics


def predict_and_save(model, test_loader, output_file):
    """Функция для предсказания и сохранения рзультатов по тестовой выборке"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:  # img_name_batch нам не нужен для индексации
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Создаем DataFrame с индексами и предсказаниями
    results_df = pd.DataFrame({
        "Id": range(len(predictions)),  # Порядковый номер объекта в датасете
        "target_feature": predictions
    })

    # Сохраняем файл в формате CSV
    results_df.to_csv(output_file, index=False)
    print(f"Предсказания сохранены в файл {output_file}")

def visualize_model_results(train_losses, val_losses, val_metrics):
    """Функция для визуализации графиков ошибок и метрик"""
    val_acc = [m[0] for m in val_metrics] 
    val_f1 = [m[3] for m in val_metrics]

    plt.figure(figsize=(12, 6)) 
    plt.subplot(1, 2, 1) 
    plt.plot(range(0, 10), train_losses, label="Train Loss")
    plt.plot(range(0, 10), val_losses, label="Val Loss")
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.title("Training Loss") 
    plt.legend() 

    plt.subplot(1, 2, 2) 
    plt.plot(val_acc, label="Validation Accuracy") 
    plt.plot(val_f1, label="Validation F1-Score") 
    plt.xlabel("Epoch") 
    plt.ylabel("Value") 
    plt.title("Validation Metrics") 
    plt.legend() 

    plt.tight_layout() 
    plt.show()


def save_log_model_results(file_name, train_losses, val_losses, val_metrics):
    """
    Сохраняет результаты обучения модели в лог-файл.

    Parameters:
        file_name (str): название файла для сохранения логов.
        train_losses (list, ndarray): список значений ошибки на обучающей выборке.
        val_lossses (list, ndarray): список значений ошибки на валидационной выборке.
        val_metrics (list, ndarray): метрики валидаций
    """

    with open(file_name, "w") as f:
        f.write("Training Results Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Epochs: {len(train_losses)}\n\n")
        
        f.write("Epoch-wise Results:\n")
        for epoch, (train_loss, val_loss, metrics) in enumerate(zip(train_losses, val_losses, val_metrics), start=1):
            acc, prec, rec, f1 = metrics
            f.write(
                f"Epoch {epoch}:\n"
                f"  Train Loss: {train_loss:.4f}\n"
                f"  Val Loss: {val_loss:.4f}\n"
                f"  Val Accuracy: {acc:.4f}\n"
                f"  Val Precision: {prec:.4f}\n"
                f"  Val Recall: {rec:.4f}\n"
                f"  Val F1-Score: {f1:.4f}\n\n"
            )
        f.write("=" * 50 + "\n")
    print(f"Лог результатов сохранен в файл {file_name}")