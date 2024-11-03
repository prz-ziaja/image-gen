from ray import tune
import torch.nn as nn
from image_gen.constants import plugin_spec, function_spec

IMAGE_SIZE = (32, 32)
DATASET_NAME = "image_gen.io.datasets.fmnist_001"

preprocessing = {
    "dataset_name": DATASET_NAME,
}

training = {
    "max_num_epochs": 1000,
    "max_num_samples": 1,
    "model_name": "image_gen.models.wgan_v2",
    "experiment_name": "Default",#"generator",
    "scaling_config": {
        "num_workers": 1,
        "resources_per_worker": {"CPU": 3},
        #"resources_per_worker": {"GPU": 0.4},
        #"use_gpu": True,
    },
    "data_module_hparams_shared":{
        "image_size": IMAGE_SIZE,
        "columns": ("image", "encoded_sentence"),
        "batch_size": 128,
        "encoded_sentence_size": 10,
    },
    "data_module_kwargs":{
        "dataset_name": DATASET_NAME,
        "reading_class": "image_gen.io.local_fs",
    },
    "hparams": {
        "lr": 0.0001,#tune.qloguniform(1e-4, 1e-2, 1e-5),
        "image_noise_t": 80,
        "noise_vector_len": 32,
        "image_ch":1,
        "T": 150,
        "image_ch": 1,
        "t_start": 1e-4,
        "t_end": 2e-2,
        "number_of_generated_batches": 2,
        "clamp_val": 1.4
    },
}
