from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. patchs.
    """
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'

    def __post_init__(self):
        # create the directory if they does not exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)


# expose a singleton
project = Project()
