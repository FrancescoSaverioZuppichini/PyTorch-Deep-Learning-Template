from typing import Union, Any
from torch.utils.data import Dataset
from torch import Tensor

class MyDataset(Dataset):
    def __getitem__(self, idx: int) -> Any:
        pass

    def __len__(self) -> int:
        return 0