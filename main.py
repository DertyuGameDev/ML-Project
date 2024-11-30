import torch
from torch import nn
from tools import get_dataloader, show_metrics, train_and_evaluate

train_dl, test_dl = get_dataloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LinerModel(torch.nn.Module):
    def __init__(self):
        super(LinerModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 20),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y

    def process(self, x):
        return self(x)


l_model = LinerModel()

show_metrics(*train_and_evaluate(l_model, train_dl, test_dl, 50))

torch.save(l_model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model2.pth")
