import torch
from pandas.core.methods.to_dict import to_dict
from torchvision import transforms
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import ToDtype

from tools import get_dataloader, show_metrics, train_and_evaluate, get_result

train_dl, test_dl = get_dataloader(transform=transforms.Compose([
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        ToDtype(torch.float32, scale=True),
                                        Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# l_model = LinerModel_v2()
#
# show_metrics(*train_and_evaluate(l_model, train_dl, test_dl, 50))
#
# torch.save(l_model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model1_v2.pth")

# test_model = torch.load("model1.pth")
#
# get_result(test_model)