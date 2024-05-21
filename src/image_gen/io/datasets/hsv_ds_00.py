from image_gen.constants import plugin_triplet
from image_gen.constants.secret import s3_secrets
from image_gen.io.dataloaders.npy_loader import customDataModule

input_path = "/ray/image_gen/raw/"
output_path = "/ray/image_gen/hsv_ds_00/"
dataloader = customDataModule

keys_to_save = [
    "hsv",
    "labels",
    "test",
]

source_loader = plugin_triplet(
    "image_gen.io.source_loader_s3",
    "ray_read_image_gen_raw",
    {"dir_path": input_path},
)

plugins = (
    plugin_triplet(
        "image_gen.preprocessing.format_converters",
        "rgb_to_hsv_map_batches",
        {"image_key": "data", "keep_source": False},
    ),
)

output_writer = plugin_triplet(
    "image_gen.io.source_loader_s3",
    "ray_write_results",
    {"columns": keys_to_save, "dir_path": output_path},
)
