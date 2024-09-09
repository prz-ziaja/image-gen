from image_gen.constants import plugin_spec, function_spec
from image_gen.io.dataloaders.npy_loader import customDataModule

input_path = "/home/przemek/Desktop/image-gen/data"
output_path = "/home/przemek/Desktop/image-gen/data/processed/"
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
    plugin_spec(
        "image_gen.preprocessing.sentence_embedding",
        "SentenceEncoder_map_batches",
        {},
        {"text_key": "caption", "result_name": "encoded_sentence", "keep_source": False},
        {},
    ),
)

output_writer = function_spec(
    "",
    "write_numpy",
    {"columns": keys_to_save, "dir_path": output_path}
)
