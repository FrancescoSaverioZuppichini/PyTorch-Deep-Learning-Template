from poutyne.framework.callbacks import Callback


from pytorch_lightning.callbacks import Callback


class MyCallback(Callback):
    """A custom callback, check here:
    
    https://pytorch-lightning.readthedocs.io/en/latest/generated/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback

    Args:
        Callback ([type]): [description]
    """

    def on_init_start(self, trainer):
        pass

    def on_init_end(self, trainer):
        pass

    def on_train_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_test_end(self, trainer, pl_module):
        pass