import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, random_split
from importlib import import_module
from pathlib import Path
import torchvision
import torchvision.transforms as transforms


class customDataset(Dataset):
    def __init__(self):
        self.train_ds = torchvision.datasets.FashionMNIST(
            "/tmp/data/",
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(32),])
        )

    def __len__(self):
        return len(self.train_ds)

    def __getitem__(self, idx):
        sample = self.train_ds[idx]
        enc = torch.zeros(10)
        enc[sample[1]] = 1
        return {"image": sample[0], "encoded_sentence":enc}
class customDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, columns: tuple=None, batch_size=512, **kwargs):
        pl.LightningDataModule.__init__(self)

        dataset = import_module(dataset_name)
        self.dataset_name = dataset_name
        
        self.columns = columns
        self.batch_size = batch_size

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        self.train_ds = customDataset()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.val_ds, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size)

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
