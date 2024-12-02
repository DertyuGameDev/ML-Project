import torch
from torch import nn
from torchvision.transforms import transforms, Normalize
from torchvision.transforms.v2 import ToDtype


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 20),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y

    def process(self, x):
        return self(x)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])
