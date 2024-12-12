import torch
from torch import optim

from torchvision import transforms
from scripts.checkpoint import load_checkpoint
from scripts.dataset_manager import get_dataloader
from scripts.trainer import train_and_evaluate
from tools import show_metrics, get_result
from torchsummary import summary

from Models.VGG import model as model_data
train_transform_2 = transforms.Compose([
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.5, p=0.5)]),  # Перспективные искажения
    transforms.ColorJitter(brightness=0.6, saturation=0.4),  # Более мягкие цветовые изменения
    transforms.RandomApply([transforms.RandomAffine(degrees=10, shear=5)]),  # Легкий сдвиг
    transforms.RandomHorizontalFlip(p=0.5),

])

train_dl, test_dl = get_dataloader(transform=train_transform_2, batch=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model_data.Model().to(device)
summary(model, (3, 128, 128))
optimizer = optim.Adam(model.parameters(), lr=1E-1)
# epoch, loss, metrics = load_checkpoint(model, optimizer, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\Models\\UnetModel_1\\checkpoints\\model_35_hist_0.30258899676375406")

show_metrics(*train_and_evaluate(model, train_dl, test_dl, optimizer, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\Models\\VGG", num_epochs=75))

torch.save(model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\Models\\VGG\\model1.pth")

# test_model = torch.load("C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model1_v2.pth")
#
# get_result(test_model, model_data.transform)