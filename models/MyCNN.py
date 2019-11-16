import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, in_features=3, classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(128, classes))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
