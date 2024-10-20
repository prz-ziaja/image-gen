from image_gen.constants import plugin_spec, function_spec
from image_gen.io.dataloaders.fmnist_loader import customDataModule
from torchvision.transforms import v2
import torch



dataloader = customDataModule

# transform = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32),
#     v2.Resize(64),
#     v2.RandomCrop((64,64)),
#     v2.Normalize(mean=[123.68, 116.28, 103.53], std=[58.4, 57.12, 57.38]),
# ])
