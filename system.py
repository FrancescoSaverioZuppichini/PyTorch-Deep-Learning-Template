from torch import Tensor
from torch import optim
import pytorch_lightning as pl
# pytorch-lighting violates avery software enginner good practices by forcing you to place
# all your shit inside they module. We, instead, used a correct module approach
# and we will import our model. Also, the LightningModule heavily force you to hardcode your stuff
# inside it, press 'f' for the DRY principle.
from models import MyCNN
from torch import nn
from pytorch_lightning.metrics.classification.accuracy import Accuracy
from pytorch_lightning.metrics.classification.f_beta import F1
from typing import Dict, Union, Callable


class MySystem(pl.LightningModule):

    def __init__(self, model: nn.Module, criterion: Callable = nn.CrossEntropyLoss, lr :float = 1e-3):
        super().__init__()
        self.model = model
        self.criterion = criterion()
        self.lr = lr
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
            'f1': F1()
        })
        # this apis are so bad
        
    def compute_metrics(self, out: Tensor, batch: Union[Tensor, Tensor]) -> Dict[str, Tensor]:
        x, y = batch
        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric(out, y)
        return metrics

    def basic_step(self, batch: Union[Tensor, Tensor], name_space: str) -> Tensor:
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        metrics = self.compute_metrics(out, batch)
        self.log_dict({f'{name_space}_loss': loss, **
                       {f'{name_space}_{k}': v for k, v in metrics.items()}})
        return loss

    def training_step(self, batch: Union[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.basic_step(batch, name_space='train')
        return loss

    def validation_step(self, batch: Union[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.basic_step(batch, name_space='val')
        return loss

    def test_step(self, batch: Union[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.basic_step(batch, name_space='test')
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
