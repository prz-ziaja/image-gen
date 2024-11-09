from image_gen.io.dataloaders.fmnist_loader import customDataModule
from torchvision.transforms import v2
import torch


dataloader = customDataModule

transform = [
    v2.ToTensor(),  # Scales data into [0,1]
    v2.RandomHorizontalFlip(),
    v2.Lambda(lambda t: (t * 2) - 1),
]

inv_transform = [
    v2.Normalize(
        mean=[
            -1,
        ]
        * 3,
        std=[
            1 / 127.5,
        ]
        * 3,
    ),
    v2.ToDtype(torch.uint8),
]
