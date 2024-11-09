import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from importlib import import_module
import torchvision
from torchvision.transforms import v2


class customDataset(Dataset):
    def __init__(self, dataset_name: str, image_size: int, train: bool):
        dataset = import_module(dataset_name)
        self.train_ds = torchvision.datasets.FashionMNIST(
            "/tmp/data/",
            download=True,
            # train=train,
            transform=v2.Compose(
                [
                    *dataset.transform,
                    v2.Resize(image_size),
                ]
            ),
        )

    def __len__(self):
        return len(self.train_ds)

    def __getitem__(self, idx):
        sample = self.train_ds[idx]
        enc = torch.zeros(10)
        enc[sample[1]] = 1
        return {"image": sample[0], "encoded_sentence": enc}


class customDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        image_size: int,
        columns: tuple = None,
        batch_size=512,
        **kwargs,
    ):
        pl.LightningDataModule.__init__(self)

        self.dataset_name = dataset_name

        self.columns = columns
        self.image_size = image_size
        self.batch_size = batch_size

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        if stage == "fit":
            train_val_ds = customDataset(
                dataset_name=self.dataset_name, image_size=self.image_size, train=True
            )
            self.train_ds, self.val_ds = random_split(train_val_ds, [0.8, 0.2])
        elif stage == "test":
            train_val_ds = customDataset(
                dataset_name=self.dataset_name, image_size=self.image_size, train=False
            )
        else:
            raise Exception(f"Stage `{stage}` is not supported - pick (fit|test)")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
