import torch
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class MyDataset(Dataset):
    """
    Custom dataset to read the images from a folder using the directory name as class label for the contained images.
    """
    images: [Image] = field(default_factory=list)
    labels: [int] = field(default_factory=list)
    labels2name: dict = None
    transform: callable = None

    def __getitem__(self, i):
        img = self.images[i]
        label = torch.tensor(self.labels[i]).long()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    @classmethod
    def from_dir(cls, root, img_format='jpg', *args, **kwargs):
        labels2name = defaultdict(str)
        labels = []
        images = []
        for i, dir in enumerate(root.iterdir()):
            label = dir.name
            images = images + list(dir.glob(f'*.{img_format}'))
            labels += ([i] * len(images))
            labels2name[i] = label
        return cls.from_paths(images, labels, labels2name, *args, **kwargs)

    @classmethod
    def from_paths(cls, img_paths, *args, **kwargs):
        images = [Image.open(img_path) for img_path in img_paths]
        return cls(images, *args, **kwargs)
