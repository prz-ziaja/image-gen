from image_gen.constants import plugin_spec, function_spec
from image_gen.io.dataloaders.npy_loader import customDataModule
from torchvision.transforms import v2
import torch

dataloader_reader_module = "image_gen.io.local_fs"
input_path = "/home/przemek/Desktop/image-gen/data"
output_path = "/home/przemek/Desktop/image-gen/data/processed/"
# in case of s3
# dataloader_reader_module = "image_gen.io.s3"
# input_path = "s3://ray/coco-2017"
# output_path = "s3://ray/coco-2017/processed/"
dataloader = customDataModule
metadata_path = output_path
image_dir_path = input_path

# transform for dataloader
transform = [
    v2.ToImage(),
    v2.ToDtype(torch.float32),
    v2.Normalize(
        mean=[
            127.5,
        ]
        * 3,
        std=[
            127.5,
        ]
        * 3,
    ),
]

# inverse transform for training validation
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

keys_to_save = [
    "file_name",
    "is_train",
    "encoded_sentence",
]

source_loader = function_spec(
    "image_gen.io.local_fs",
    # in case of s3
    # "image_gen.io.s3",
    "read_metadata",
    {"coco_path": input_path},
)

ray_source_connector = function_spec(
    "ray.data",
    "from_pandas",
    {"override_num_blocks": 4},
)

plugins = (
    plugin_spec(
        "image_gen.preprocessing.sentence_embedding",
        "SentenceEncoder_map_batches",
        {
            "text_key": "caption",
            "result_name": "encoded_sentence",
            "keep_source": False,
        },
        {"SENETENCE_ENCODER_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2"},
        {"num_gpus": 0.25, "concurrency": 4, "batch_size": 128},
    ),
)

output_writer = function_spec(
    "",
    "write_numpy",
    {
        "column": keys_to_save,
        "path": output_path,
        # in case of s3
        # "filesystem": get_s3_fs_pa()
    },
)
