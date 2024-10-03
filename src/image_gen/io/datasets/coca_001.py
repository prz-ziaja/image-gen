from image_gen.constants import plugin_spec, function_spec
from image_gen.io.dataloaders.npy_loader import customDataModule
from torchvision.transforms import v2
import torch

dataloader_reader_module = "image_gen.io.local_fs"
input_path = "/home/przemek/Desktop/image-gen/data"
output_path = "/home/przemek/Desktop/image-gen/data/processed/"
dataloader = customDataModule
metadata_path = output_path
image_dir_path = input_path

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32),
    v2.Resize(128),
    v2.RandomCrop((128,128)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

keys_to_save = [
    "file_name",
    "is_train",
    "encoded_sentence",
]

source_loader = function_spec(
    "image_gen.io.local_fs",
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
        {"text_key": "caption", "result_name": "encoded_sentence", "keep_source": False},
        {"SENETENCE_ENCODER_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2"},
        {"concurrency": 4},
    ),
)

output_writer = function_spec(
    "",
    "write_numpy",
    {"column": keys_to_save, "path": output_path}
)
