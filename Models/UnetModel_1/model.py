import torch
from torch import nn
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype


class Conv(nn.Module):
    def __init__(self, in_num_channels, out_num_channels):
        super(Conv, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_num_channels, out_num_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_num_channels),

            nn.Conv2d(out_num_channels, out_num_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_num_channels)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class DConv(nn.Module):
    def __init__(self, in_num_channels, out_num_channels):
        super(DConv, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_num_channels, out_num_channels, (2, 2), stride=2, padding=0, bias=True)
        self.l2 = Conv(in_num_channels, out_num_channels)

    def forward(self, x, y):
        x1 = self.l1(x)
        data = torch.cat((x1, y), 1)
        x2 = self.l2(data)
        return x2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Энкодер
        self.e1 = Conv(3, 64)
        self.e2 = Conv(64, 128)
        # self.e3 = Conv(128, 256)
        # self.e4 = Conv(16, 32)
        self.trans = Conv(128, 256)
        self.max_pool = MaxPool2d((2, 2), stride=2)


        # Декодер
        # self.d1 = DConv(1024, 512)
        # self.d2 = DConv(512, 256)
        self.d3 = DConv(256, 128)
        self.d4 = DConv(128, 64)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 20, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # ЭНКОДЕР
        x_e1 = self.e1(x) # 64x512x512
        x_e2 = self.e2(self.max_pool(x_e1))  # 128x256x256
        # x_e3 = self.e3(self.max_pool(x_e2))  # 256x128x128
        # x_e4 = self.e4(self.max_pool(x_e3))  # 512x64x64

        x_trans = self.trans(self.max_pool(x_e2)) # 1024x32x32

        # ДЕКОДЕР
        # x_d1 = self.d1(x_trans, x_e4)
        # x_d2 = self.d2(x_d1, x_e3)
        x_d3 = self.d3(x_trans, x_e2)
        x_d4 = self.d4(x_d3, x_e1)

        ans = self.classifier(x_d4)
        ans = ans.view(ans.size(0), -1)
        # print(ans.shape)
        # ans = torch.argmax(ans, dim=1).type(torch.float32)
        # ans.requires_grad = True
        # print(ans.shape)
        # print(ans)

        return ans

    def process(self, x):
        return self(x)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ToDtype(torch.float32, scale=True),
    Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])