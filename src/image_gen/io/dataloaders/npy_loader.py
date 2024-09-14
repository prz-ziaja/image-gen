import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, random_split
from importlib import import_module


class customDataset(Dataset):
    def __init__(self, image_dir_path:str, data: dict, columns: tuple, reader, transform=None):
        self.columns = columns
        self.metadata = data
        self.image_dir_path = image_dir_path
        self.transform = transform
        self.reader = reader

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        return None

class customDataModule(pl.LightningDataModule):
    def __init__(self, image_dir_path:str, metadata_path: str, columns: tuple, reading_class:str, batch_size=32):
        pl.LightningDataModule.__init__(self)
        self.metadata_path = metadata_path
        self.columns = columns
        self.batch_size = batch_size
        self.image_dir_path = image_dir_path
        self.reader = import_module(reading_class).Reader()

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        loaded_data = self.Reader.read_metadata(self.metadata_path)
        if stage == "fit":
            train_val_ds = customDataset(loaded_data, self.columns, test=False)
            self.train_ds, self.val_ds = random_split(train_val_ds, [0.8, 0.2])
        elif stage == "test":
            self.test_ds = customDataset(loaded_data, self.columns, test=True)
        else:
            raise Exception(f"{stage} not supported")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
