from enum import Enum, auto
import glob
import importlib
import pathlib
from typing import Type

from torch.utils.data import Dataset

available_datasets: set[Type[Dataset]] = set()
available_validation_datasets: set[Type[Dataset]] = set()


class DatasetType(Enum):
    LABELLED_DATASET = auto()
    UNLABALLED_DATASET = auto()


mapping = {
    DatasetType.LABELLED_DATASET: available_validation_datasets,
    DatasetType.UNLABALLED_DATASET: available_datasets,
}


def register_dataset(type: DatasetType):
    def inner(cls):
        mapping[type].add(cls)
        return cls
    return inner


# dynamically load sibling modules
potential_data_modules = [
    pathlib.Path(module).name.replace('.py', '')
    for module in glob.glob(f'{pathlib.Path(__file__).parent}/*.py')
    if module != '__init__.py'
]
for module in potential_data_modules:
    importlib.import_module(f'src.data.{module}')
