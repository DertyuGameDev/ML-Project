from pandas.core.methods.to_dict import to_dict

from models import *
from tools import get_dataloader, show_metrics, train_and_evaluate, get_result

train_dl, test_dl = get_dataloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

l_model = LinerModel()

# show_metrics(*train_and_evaluate(l_model, train_dl, test_dl, 50))
#
# torch.save(l_model, "C:\\Users\\kosty\\gadflhahjadgjtma\\ML-Project\\model2.pth")

test_model = torch.load("model1.pth")

get_result(test_model)