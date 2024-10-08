import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, random_split
from importlib import import_module
from pathlib import Path

"""
Structure of directory containing images
 - directory
   - validation (test dataset)
     - data
       - *.jpg
   - train (train dataset)
     - data
       - *.jpg
"""

class customDataset(Dataset):
    def __init__(self, image_dir_path:str, metadata: dict, columns: tuple, dataset_name:str, reading_class:str):
        self.columns = columns
        self.metadata = metadata
        self.image_dir_path = Path(image_dir_path)

        dataset = import_module(dataset_name)
        self.transform = dataset.transform
        self.reader = import_module(reading_class).Reader()

    def __len__(self):
        return len(self.metadata['file_name'])

    def __getitem__(self, idx):
        file_name = self.metadata['file_name'][idx]
        subdir = "train" if self.metadata['is_train'][idx] else "validation"
        loaded_image = self.reader.read_image(str(self.image_dir_path / subdir / "data" / file_name))
        image = self.transform(loaded_image) if self.transform is not None else loaded_image
        return {k:self.metadata.get(k, None) for k in self.columns} | {'image': image}

class customDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, reading_class:str, columns: tuple=None, batch_size=512, **kwargs):
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
        assert "is_train" in loaded_data and "file_name" in loaded_data
        train = {k:v[loaded_data['is_train']] for k,v in loaded_data.items()}
        test = {k:v[~loaded_data['is_train']] for k,v in loaded_data.items()}
        if stage == "fit":
            train_val_ds = customDataset(image_dir_path=self.image_dir_path, metadata=train, columns=self.columns, dataset_name=self.dataset_name, reading_class=self.reading_class)
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

if __name__ == "__main__":
    import sys
    import torch.nn as nn
    sys.path.append("/home/przemek/Desktop/image-gen/src")
    c = nn.Conv2d(3,5,3)
    dd = customDataModule('image_gen.io.datasets.coca_001', 'image_gen.io.local_fs', ['encoded_sentence', ], batch_size=8)
    dd.setup("test")
    test_dl = dd.test_dataloader()
    for i in test_dl:
        print(c(i['image']).shape)
        break
