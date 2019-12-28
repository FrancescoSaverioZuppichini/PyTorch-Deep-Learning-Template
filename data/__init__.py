import numpy as np
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder
from torchbearer.cv_utils import DatasetValidationSplitter

def get_dataloaders(
        root,
        train_transform=None,
        val_transform=None,
        val_split=0.2,
        test_split=0.5,
        batch_size=64):
    """
    This function returns train, val and test dataloaders.
    """
    # create the datasets
    ds = ImageFolder(root=root, transform=val_transform)
    splitter = DatasetValidationSplitter(len(ds), val_split)
    train_ds = splitter.get_train_dataset(ds)
    train_ds.dataset.transform = train_transform
    val_ds = splitter.get_val_dataset(ds)
    test_ds = val_ds
    if test_split is not None:
        # let's further split the val dataset in val and test
        splitter = DatasetValidationSplitter(len(val_ds), test_split)
        val_ds = splitter.get_train_dataset(val_ds.dataset)
        test_ds = splitter.get_val_dataset(val_ds.dataset)

    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    return train_dl, val_dl, test_dl