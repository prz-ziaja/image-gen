from image_gen.constants import plugin_spec, function_spec
from image_gen.io.dataloaders.npy_loader import customDataModule

input_path = "/ray/image_gen/raw/"
output_path = "/ray/image_gen/hsv_ds_00/"
dataloader = customDataModule

keys_to_save = [
    "filename",
    "is_train",
    "encoded_sentence",
]

source_loader = function_spec(
    "image_gen.io.local_fs",
    "read_metadata",
    {"dir_path": input_path},
)

ray_source_connector = function_spec(
    "ray.data",
    "from_pandas",
    {"override_num_blocks": 4},
)

plugins = (
    plugin_quadruplet(
        "image_gen.preprocessing.format_converters",
        "rgb_to_hsv_map_batches",
        {"image_key": "data", "keep_source": False},
        {},
    ),
)

output_writer = plugin_quadruplet(
    "",
    "write_numpy",
    {"columns": keys_to_save, "dir_path": output_path},
    {},
)
