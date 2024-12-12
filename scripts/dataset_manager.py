import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype

import constants


base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])


class CustomDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data_frame = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # print(self.data_frame[idx])
        img_name = os.path.join(self.root_dir,
                                str(self.data_frame[idx][0]) + ".jpg")  # ID изображений в первом столбце
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame[idx][1]  # Метки во втором столбце

        if self.transform:
            image = self.transform(image)
        image = base_transform(image)
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


def get_dataloader(ratio: tuple[float, float] = (0.95, 0.05),
                    csv_file: str = constants.csv_train_ans,
                    root_dir: str = constants.root_dir_train,
                    batch: int = 128,
                    transform=None,
                   ) -> tuple:
    data = pd.read_csv(csv_file).values
    train_data, test_data = random_split(data, ratio, torch.Generator().manual_seed(1))
    train_ds = CustomDataset(data=train_data, root_dir=root_dir, transform=transform)
    test_ds = CustomDataset(data=test_data, root_dir=root_dir, transform=None)
    return DataLoader(train_ds, batch_size=batch, shuffle=True), DataLoader(test_ds, batch_size=128)