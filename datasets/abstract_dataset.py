from abc import ABC, abstractmethod
import torch
from torch_geometric.data import InMemoryDataset

class AbstractDatasetInfos(ABC):
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    @abstractmethod
    def compute_input_output_dims(self, datamodule):
        pass

class AbstractDataModule(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.training_iterations = None

    @abstractmethod
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader 