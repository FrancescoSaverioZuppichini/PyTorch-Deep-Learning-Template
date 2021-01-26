import json
import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from models import MyCNN, resnet18
from utils import device, show_dl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, MLFlowLogger
from system import MySystem
from logger import logging

if __name__ == '__main__':
    project = Project()

    with open('./secrets.json', 'r') as f:
        secrets = json.load(f)

    seed_everything(0)
    # our hyperparameters
    params = {
        'lr': 1e-3,
        'batch_size': 128,
        'epochs': 10,
        'model': 'resnet18-finetune',
        'id': time.time()
    }
    logging.info(f'Using device={device} 🚀')
    # everything starts with the data
    train_dl, val_dl, test_dl = get_dataloaders(
        project.data_dir / "tiny-imagenet-200",
        val_transform=val_transform,
        train_transform=train_transform,
        batch_size=params['batch_size'],
        pin_memory=True,
        num_workers=4,
    )
    # is always good practice to visualise some of the train and val images to be sure data-aug
    # is applied properly
    # show_dl(train_dl)
    # show_dl(test_dl)
    # create our special resnet18
    cnn = resnet18(200).to(device)
    # print the model summary to show useful information
    logging.info(summary(cnn, (3, 224, 244)))
    # define and create the model's chekpoints dir
    model_checkpoint_dir = project.checkpoint_dir / str(params['id'])
    model_checkpoint_dir.mkdir(exist_ok=True)
    # our callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=model_checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.2f}'
        ),
        EarlyStopping(monitor='val_loss', patience=10, verbose=True)
    ]
    # using commet
    logger = CometLogger(
        api_key=secrets['COMET_API_KEY'],
        project_name="test",
        workspace="francescosaveriozuppichini"
    )
    logger.log_hyperparams(params)
    
    system = MySystem(model=cnn, lr=params['lr'])

    trainer = Trainer(gpus=1, min_epochs=params['epochs'],
                      progress_bar_refresh_rate=20, logger=logger,
                      callbacks=callbacks)

    trainer.fit(system, train_dl, val_dl)

    print(trainer.test(test_dataloaders=test_dl))
