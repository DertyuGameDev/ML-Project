import time

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from tqdm import tqdm

from scripts import constants
from scripts.checkpoint import save_checkpoint
from scripts.tools import save_hist


def train_and_evaluate(model, train_loader, val_loader, num_epochs=30, lr=1E-1):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    val_metrics = []

    for epoch in range(num_epochs):
        # Обучение
        train_loss = run_epoch(loss_func, model, optimizer, train_loader)
        train_losses.append(train_loss)

        # Оценка
        all_labels, all_preds, val_loss = ran_val_epoch(loss_func, model, val_loader)
        val_losses.append(val_loss)

        # подсчёт метрик
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        val_metrics.append((accuracy, precision, recall, f1))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {accuracy:.4f}, Val Prec: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")
        if epoch % 5 == 4:
            save_checkpoint(model, optimizer, epoch, val_loss, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\Models\\UnetModel_1\\checkpoints\\" + "model_" + str(epoch + 1) + "_" + str(accuracy))
        time.sleep(1)

        save_hist(val_metrics, train_losses, val_losses, constants.hist_path)
    return train_losses, val_losses, val_metrics


def run_epoch(loss_func, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(constants.device), labels.to(constants.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def ran_val_epoch(loss_func, model, val_loader):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(constants.device), labels.to(constants.device)
            images, labels = images, labels
            outputs = model(images)
            loss = loss_func(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return all_labels, all_preds, val_loss