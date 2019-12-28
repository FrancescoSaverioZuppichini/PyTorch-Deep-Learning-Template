import torchbearer
from torchbearer.callbacks import Callback

class CometCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_end_epoch(self, state):
        self.experiment.log_metrics(state[torchbearer.METRICS])
