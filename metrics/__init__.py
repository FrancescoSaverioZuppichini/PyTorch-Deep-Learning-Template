import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.utils import _input_format_classification

class MyAccuracy(Metric):
    """Custom metric, check here:

    https://pytorch-lightning.readthedocs.io/en/latest/metrics.html?highlight=metric#metric-api
    
    Args:
        Metric ([type]): [description]
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # we maybe have to preprocess
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total