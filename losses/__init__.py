import torch

# define custom losses
def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss

# or just define a normal PyTorch loss
def my_mse(y_pred, y_true):
   return torch.mean((y_pred - y_true) ** 2)