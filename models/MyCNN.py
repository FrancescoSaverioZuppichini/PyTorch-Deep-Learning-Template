import torch
import torch.nn as nn
from .blocks import ConvBnAct


class MyCNN(nn.Module):
    def __init__(self, in_features: int = 3, classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnAct(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            ConvBnAct(1, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            ConvBnAct(1, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x
