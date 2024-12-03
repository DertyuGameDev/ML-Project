import torch

from scripts.dataset_manager import get_dataloader
from scripts.trainer import train_and_evaluate
from tools import show_metrics, get_result

from Models.UnetModel_1 import model as model_data
train_dl, test_dl = get_dataloader(transform=model_data.transform, batch=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model_data.Model()

show_metrics(*train_and_evaluate(model, train_dl, test_dl, 5, lr=1E-1))

torch.save(model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\Models\\UnetModel_1\\model.pth")

# test_model = torch.load("C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model1_v2.pth")
#
# get_result(test_model, model_data.transform)