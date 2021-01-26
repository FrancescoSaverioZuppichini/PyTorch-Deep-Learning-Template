import numpy as np
from numpy.lib.shape_base import dstack
from torchvision.datasets.folder import ImageFolder
from .MyDataset import MyDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets import ImageNet

def get_dataloaders(
        root,
        train_transform=None,
        val_transform=None,
        split=(0.8, 0.2),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    ds = ImageFolder(root=root / 'train', transform=train_transform)
    test_ds = ImageFolder(root=root / 'val', transform=val_transform)
    # now we want to split the train_ds in train and val
    val_len = int(len(ds) * split[1])
    train_ds, val_ds = random_split(ds, lengths=[len(ds) - val_len, val_len])
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl
