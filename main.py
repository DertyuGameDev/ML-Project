import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]) + ".jpg")  # Предполагаем, что путь к изображению в первом столбце
        image = Image.open(img_name).convert('RGB')  # Открываем изображение и конвертируем в RGB
        label = self.data_frame.iloc[idx, 1]  # Предполагаем, что метка во втором столбце

        if self.transform:
            image = self.transform(image)

        return image, label


csv_file = 'C:/Users/kosty/gadflhahjadgjtma/ML-Project/human_poses_data/train_answers.csv'
root_dir = 'C:/Users/kosty/gadflhahjadgjtma/ML-Project/human_poses_data/img_train'


transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])


dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
