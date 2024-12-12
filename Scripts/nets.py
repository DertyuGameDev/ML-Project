import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleUNet, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )

        # Классификатор 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 128 * 128 // 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x