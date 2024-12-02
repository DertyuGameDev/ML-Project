import torch

from scripts.dataset_manager import get_dataloader
from tools import show_metrics, get_result

from Models.LinerModel_1 import model
train_dl, test_dl = get_dataloader(transform=model.transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.Model()
#
# show_metrics(*train_and_evaluate(l_model, train_dl, test_dl, 50))
#
# torch.save(l_model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model1_v2.pth")

# test_model = torch.load("model1.pth")
#
# get_result(test_model)