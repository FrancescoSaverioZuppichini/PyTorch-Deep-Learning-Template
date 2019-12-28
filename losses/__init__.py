import torch
import torchbearer
from torchbearer import callbacks

# we can use the callbacks APIS
@callbacks.add_to_loss
def l1_norm(state):
   loss = 0
   for param in state[torchbearer.MODEL].parameters():
      loss += torch.sum(torch.abs(param))
   return loss

# or just define a normal PyTorch loss
def my_mse(y_pred, y_true):
   return torch.mean((y_pred - y_true) ** 2)