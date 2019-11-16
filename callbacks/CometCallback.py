from poutyne.framework.callbacks import Callback


class CometCallback(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs):
        self.experiment.log_metrics(logs)
