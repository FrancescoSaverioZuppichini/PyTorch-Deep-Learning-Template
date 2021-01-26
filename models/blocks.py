from typing import OrderedDict
from torch import nn

class ConvBnAct(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(OrderedDict({
            'conv': nn.Conv2d(in_features, out_features, **kwargs),
            'bn': nn.BatchNorm2d(out_features),
            'act': nn.ReLU()
        }))