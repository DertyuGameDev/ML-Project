import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pygame.draw_py import draw_aaline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import csv


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx // 2, 0]) + ".jpg")  # ID изображений в первом столбце
            image = Image.open(img_name).convert('RGB')
            label = self.data_frame.iloc[idx // 2, 1]  # Метки во втором столбце
        else:
            img_name = os.path.join(self.root_dir,
                                    str(self.data_frame.iloc[idx // 2, 0]) + ".jpg")  # ID изображений в первом столбце
            image = Image.open(img_name).convert('RGB')
            image.transpose(Image.FLIP_LEFT_RIGHT)
            label = self.data_frame.iloc[idx // 2, 1]  # Метки во втором столбце

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_frame = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame[idx]))
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame[idx].replace(".jpg", '')


        if self.transform:
            image = self.transform(image)

        return image, int(label)


def get_dataloader(ratio: tuple[float, float] = (0.9, 0.1),
                    csv_file: str = 'C:/Users/kosty/gadflhahjadgjtma/ML-Project/human_poses_data/train_answers.csv',
                    root_dir: str = 'C:/Users/kosty/gadflhahjadgjtma/ML-Project/human_poses_data/img_train',
                    transform=transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        ToDtype(torch.float32, scale=True),
                        Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
                   ) -> tuple:
    train_ds, test_ds = random_split(CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform), ratio)
    return DataLoader(train_ds, batch_size=128, shuffle=True), DataLoader(test_ds, batch_size=128, shuffle=True)


def train_and_evaluate(model, train_loader, val_loader, num_epochs=30, lr=1E-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    val_metrics = []

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
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
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                images, labels = images, labels
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average="weighted")
        rec = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        val_metrics.append((acc, prec, rec, f1))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Val Prec: {prec:.4f}, Val Recall: {rec:.4f}, Val F1: {f1:.4f}")
        time.sleep(1)
    return train_losses, val_losses, val_metrics


def show_metrics(train_loss_hist, test_loss_hist, test_metrics):
    clear_output()

    plt.figure(figsize=(12, 12))

    # График потерь
    plt.subplot(3, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.title('Test Loss')
    plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.title('Test Accuracy')
    plt.plot(np.arange(len(test_metrics)), list(map(lambda x: x[0],test_metrics)))
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.title('Test F1 score')
    plt.plot(np.arange(len(test_metrics)), list(map(lambda x: x[3],test_metrics)))
    plt.grid()

    plt.tight_layout()
    plt.show()


def get_result(model: torch.nn.Module):
    dataset = CustomTestDataset("/human_poses_data/img_test",
                                transform=transforms.Compose([
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        ToDtype(torch.float32, scale=True),
                                        Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]))
    dl = DataLoader(dataset, batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ans = []
    for img, label in tqdm(dl):
        img = img.to(device)
        pred = model.to(device)(img)
        preds = torch.argmax(pred, dim=1)
        res = torch.cat((torch.tensor(label).unsqueeze(1), preds.unsqueeze(1)), dim=1)
        ans.extend(res)
    ans = [[element.item() for element in row] for row in ans]
    with open('../result.csv', 'w', newline="") as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(['id', 'target_feature'])
        writer.writerows(ans)

