import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
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
    def __init__(
        self,
        image_dir_path: str,
        metadata: dict,
        columns: tuple,
        image_size: int,
        dataset_name: str,
        reading_class: str,
    ):
        self.columns = columns
        self.metadata = metadata
        self.image_dir_path = Path(image_dir_path)

        dataset = import_module(dataset_name)
        self.transform = v2.Compose(
            [
                *dataset.transform,
                v2.Resize(image_size),
                v2.RandomCrop(image_size),
            ]
        )
        self.reader = import_module(reading_class).Reader()
        for column in columns:
            if column not in self.metadata and column != "image":
                raise Exception(
                    f"Column {column} not present in metadata. Please remove column or add column to metadata dataset."
                )

    def __len__(self):
        return len(self.metadata["file_name"])

    def __getitem__(self, idx):
        file_name = self.metadata["file_name"][idx]
        subdir = "train" if self.metadata["is_train"][idx] else "validation"
        loaded_image = self.reader.read_image(
            str(self.image_dir_path / subdir / "data" / file_name)
        )
        image = (
            self.transform(loaded_image) if self.transform is not None else loaded_image
        )
        return {
            k: self.metadata[k][idx] for k in self.columns if k in self.metadata
        } | {"image": image}


class customDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        reading_class: str,
        image_size: int,
        columns: tuple = None,
        batch_size=512,
        num_workers=4,
        **kwargs,
    ):
        pl.LightningDataModule.__init__(self)

        dataset = import_module(dataset_name)
        self.dataset_name = dataset_name
        self.metadata_path = dataset.metadata_path
        self.image_dir_path = dataset.image_dir_path

        self.columns = columns
        self.batch_size = batch_size
        self.image_size = image_size

        self.reading_class = reading_class
        self.reader = import_module(reading_class).Reader()
        self.num_workers = num_workers

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        loaded_data = self.reader.read_metadata(self.metadata_path)
        assert "is_train" in loaded_data and "file_name" in loaded_data
        train = {k: v[loaded_data["is_train"]] for k, v in loaded_data.items()}
        test = {k: v[~loaded_data["is_train"]] for k, v in loaded_data.items()}
        if stage == "fit":
            train_val_ds = customDataset(
                image_dir_path=self.image_dir_path,
                metadata=train,
                columns=self.columns,
                dataset_name=self.dataset_name,
                reading_class=self.reading_class,
                image_size=self.image_size,
            )
            self.train_ds, self.val_ds = random_split(train_val_ds, [0.8, 0.2])
        elif stage == "test":
            self.test_ds = customDataset(
                self.image_dir_path,
                test,
                self.columns,
                self.dataset_name,
                self.reading_class,
                image_size=self.image_size,
            )
        else:
            raise Exception(f"Stage `{stage}` is not supported - pick (fit|test)")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
