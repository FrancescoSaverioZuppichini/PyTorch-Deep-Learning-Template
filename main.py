import time
from comet_ml import Experiment
import torchbearer
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from models import MyCNN, resnet18
from utils import device, show_dl
from torchbearer import Trial
from torchbearer.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from callbacks import CometCallback
from logger import logging

if __name__ == '__main__':
    project = Project()
    # our hyperparameters
    params = {
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 1,
        'model': 'resnet18-finetune',
        'id': time.time()
    }

    logging.info(f'Using device={device} 🚀')
    # everything starts with the data
    train_dl, val_dl, test_dl = get_dataloaders(
        project.data_dir,
        val_transform=val_transform,
        train_transform=train_transform,
        batch_size=params['batch_size'],
    )
    # is always good practice to visualise some of the train and val images to be sure data-aug
    # is applied properly
    # show_dl(train_dl)
    # show_dl(test_dl)
    # define our comet experiment
    experiment = Experiment(api_key='8THqoAxomFyzBgzkStlY95MOf',
                            project_name="dl-pytorch-template", workspace="francescosaveriozuppichini")
    experiment.log_parameters(params)
    # create our special resnet18
    cnn = resnet18(2).to(device)
    loss = nn.CrossEntropyLoss()
    # print the model summary to show useful information
    logging.info(summary(cnn, (3, 224, 244)))
    # define custom optimizer and instantiace the trainer `Model`
    optimizer = optim.Adam(cnn.parameters(), lr=params['lr'])
    # create our Trial object to train and evaluate the model
    trial = Trial(cnn, optimizer, loss, metrics=['acc', 'loss'],
                  callbacks=[
                    #   CometCallback(experiment),
                    #   ReduceLROnPlateau(monitor='val_loss',
                    #                     factor=0.1, patience=5),
                    #   EarlyStopping(monitor='val_acc', patience=5, mode='max'),
                    #   CSVLogger(str(project.checkpoint_dir / 'history.csv')),
                    #   ModelCheckpoint(str(project.checkpoint_dir / f'{params["id"]}-best.pt'), monitor='val_acc', mode='max')
    ]).to(device)
    trial.with_generators(train_generator=train_dl,
                          val_generator=val_dl, test_generator=test_dl)
    history = trial.run(epochs=params['epochs'], verbose=1)
    logging.info(history)
    preds = trial.evaluate(data_key=torchbearer.TEST_DATA)
    logging.info(f'test preds=({preds})')
    # experiment.log_metric('test_acc', test_acc)
