import PIL
import torch
from unittest import TestCase
from data import MyDataset
from Project import project


class TestMyDataset(TestCase):
    def setUp(self):
        self.dataset = MyDataset

    def test_from_dir(self):
        root = project.data_dir / 'val'
        dataset = MyDataset.from_dir(root)
        files_len = len(list(root.glob('**/*.jpg')))
        self.assertEqual(files_len, len(dataset))
        img, label = dataset[0]
        self.assertTrue(type(img) is PIL.JpegImagePlugin.JpegImageFile)
        self.assertTrue(type(label) is torch.Tensor)
