import sys
sys.path.append("/home/przemek/Desktop/image-gen/src")
import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, random_split
import image_gen.models.utils.image_diffusers as image_diffusers
from importlib import import_module
from pathlib import Path
import torchvision
from torchvision.transforms import v2
import cv2


class customDataset(Dataset):
    def __init__(self, dataset_name:str, image_size:int, train:bool, **config):
        dataset = import_module(dataset_name)
        print(f"@@@@@@@@@@@@@@@@ {config}")
        self.image_diffuser = image_diffusers.imageDiffuser(config["T"], config["t_start"], config["t_end"])
        self.image_noise_t = config["image_noise_t"]
        self.train_ds = torchvision.datasets.FashionMNIST(
            "/tmp/data/",
            download=True,
            #train=train,
            transform=v2.Compose([
                *dataset.transform,
                v2.Resize(image_size),
            ])
        )

    def __len__(self):
        return len(self.train_ds)

    def __getitem__(self, idx):
        sample = self.train_ds[idx]

        enc = torch.zeros(10)
        enc[sample[1]] = 1

        t = torch.ones([sample[0].shape[0],1], dtype=torch.long)* self.image_noise_t
        img_noisy, _ = self.image_diffuser(sample[0][None], t)
        img_noisy_np = img_noisy[0].permute([1,2,0]).numpy()
        img_noisy_np_0_1 = (img_noisy_np - img_noisy_np.min())/(img_noisy_np.max()-img_noisy_np.min())
        to_denoise = (img_noisy_np_0_1*255).astype(np.uint8)
        denoised_np = cv2.fastNlMeansDenoising(to_denoise, None,10,10,21)
        denoised = torch.tensor(denoised_np/255, dtype=torch.float32)[None]
        return {"image": sample[0], "image_noisy": img_noisy, "filtered_low_pass": denoised, "encoded_sentence":enc}

class customDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, image_size:int, columns: tuple=None, batch_size=512, **kwargs):
        pl.LightningDataModule.__init__(self)

        self.dataset_name = dataset_name
        
        self.columns = columns
        self.image_size = image_size
        self.batch_size = batch_size
        print(f"!!!!!!!!!!!!!!!!!! {kwargs}")
        self.kwargs = kwargs

    def setup(self, stage: str):
        torch.random.manual_seed(10)
        print(f"#############3 {self.kwargs}")
        if stage == "fit":
            train_val_ds = customDataset(dataset_name=self.dataset_name, image_size=self.image_size, train=True, **self.kwargs)
            self.train_ds, self.val_ds = random_split(train_val_ds, [0.8, 0.2])
        elif stage == "test":
            self.test_ds = customDataset(dataset_name=self.dataset_name, image_size=self.image_size, train=False, **self.kwargs)
        else:
            raise Exception(f"Stage `{stage}` is not supported - pick (fit|test)")

        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

if __name__ == "__main__":
    import sys
    import torch.nn as nn

    c = nn.Conv2d(1,5,3)
    dd = customDataModule('image_gen.io.datasets.fmnist_001', 64, ['encoded_sentence', ], batch_size=8, T=150, t_start=2e-4, t_end=4e-2, image_noise_t=80)
    dd.setup("test")
    test_dl = dd.test_dataloader()
    for i in test_dl:
        print(c(i['image']).shape)
        break
