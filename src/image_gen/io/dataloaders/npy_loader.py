import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, random_split
from importlib import import_module


class customDataset(Dataset):
    def __init__(self, image_dir_path:str, data: dict, columns: tuple, dataset_name:str, reading_class:str):
        self.columns = columns
        self.metadata = data
        self.image_dir_path = image_dir_path

        dataset = import_module(dataset_name)
        self.transforms = dataset.transforms
        self.reader = import_module(reading_class).Reader()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        return None

class customDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, reading_class:str, columns: tuple, batch_size=32):
        pl.LightningDataModule.__init__(self)

        dataset = import_module(dataset_name)
        self.dataset_name = dataset_name
        self.metadata_path = dataset.metadata_path
        self.image_dir_path = dataset.image_dir_path
        
        self.columns = columns
        self.batch_size = batch_size

        self.reading_class = reading_class
        self.reader = import_module(reading_class).Reader()

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        loaded_data = self.reader.read_metadata(self.metadata_path)
        train = {k:v[~loaded_data['is_train']] for k,v in loaded_data.items()}
        test = {k:v[loaded_data['is_train']] for k,v in loaded_data.items()}
        if stage == "fit":
            train_val_ds = customDataset(self.image_dir_path, train, self.columns, self.dataset_name, self.reading_class)
            self.train_ds, self.val_ds = random_split(train_val_ds, [0.8, 0.2])
        elif stage == "test":
            self.test_ds = customDataset(self.image_dir_path, test, self.columns, self.dataset_name, self.reading_class)
        else:
            raise Exception(f"Stage `{stage}` is not supported - pick (train|test)")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
