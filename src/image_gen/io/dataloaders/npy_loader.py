import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, IterableDataset, random_split

from image_gen.constants.secret import s3_secrets
from image_gen.io.s3 import read_dir


class customDataset(IterableDataset):
    def __init__(self, metadata: dict, columns: tuple, test: bool):
        self.columns = columns
        self.test = test
        self.data = dict()

        self.test_mask = (
            np.concatenate([data["test"] for data in loaded_data]) == self.test
        )

        for column in self.columns:
            self.data[column] = np.concatenate([data[column] for data in loaded_data])[
                self.test_mask
            ]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))


class customDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, columns: tuple, batch_size=32):
        pl.LightningDataModule.__init__(self)
        self.dataset_path = dataset_path
        self.columns = columns
        self.batch_size = batch_size

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        loaded_data = read_dir(self.dataset_path)
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
