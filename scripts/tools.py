import numpy as np
import torch
import constants
from torchvision import transforms
from torch.utils.data import DataLoader
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import csv

from scripts.dataset_manager import CustomTestDataset


def show_metrics(train_loss_hist, test_loss_hist, test_metrics):
    clear_output()

    plt.figure(figsize=(12, 12))

    show_metric(train_loss_hist, 'Train Loss', 1)
    show_metric(test_loss_hist, 'Test Loss', 2)
    show_metric(list(map(lambda x: x[0], test_metrics)), 'Test Accuracy', 3)
    show_metric(list(map(lambda x: x[3], test_metrics)), 'Test F1 score', 4)

    plt.tight_layout()
    plt.show()


def show_metric(hist, name, Id):
    plt.subplot(3, 2, Id)
    plt.title(name)
    plt.plot(np.arange(len(hist)), hist)
    plt.yscale('log')
    plt.grid()


def get_result(model: torch.nn.Module, transform: transforms.Compose):
    dataset = CustomTestDataset(constants.root_dir_test, transform=transform)
    dl = DataLoader(dataset, batch_size=128)

    model.eval()
    ans = []
    for img, label in tqdm(dl):
        img = img.to(constants.device)
        pred = model.to(constants.device)(img)
        preds = torch.argmax(pred, dim=1)
        res = torch.cat((torch.tensor(label).unsqueeze(1), preds.unsqueeze(1)), dim=1)
        ans.extend(res)
    ans = [[element.item() for element in row] for row in ans]
    with open('../result.csv', 'w', newline="") as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(['id', 'target_feature'])
        writer.writerows(ans)


def save_hist(test_metrics, train_loss, test_loss, path):
    with open(path, 'w') as file:
        file.writelines([" ".join(list(map(str, train_loss))), '\n',
                         " ".join(list(map(str, test_loss))), '\n',
                         " ".join(list(map(str, map(lambda x: x[0], test_metrics)))), '\n',
                         " ".join(list(map(str, map(lambda x: x[1], test_metrics)))), '\n',
                         " ".join(list(map(str, map(lambda x: x[2], test_metrics)))), '\n',
                         " ".join(list(map(str, map(lambda x: x[3], test_metrics))))])
