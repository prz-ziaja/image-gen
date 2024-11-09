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
    "model_name": "image_gen.models.wgan_v3",
    "experiment_name": "FMNIST GAN FILTERED",
    "scaling_config": {
        "num_workers": 1,
        #"resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.4},
        "use_gpu": True,
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
        "T": 150,
        "image_noise_t": 30,
        "t_start": 2e-4,
        "t_end": 4e-2,
    },
    "hparams": {
        "critic_lr": 0.00001,#tune.qloguniform(1e-4, 1e-2, 1e-5),
        "generator_lr": 0.00008,#tune.qloguniform(1e-4, 1e-2, 1e-5),
        "noise_vector_len": 64,
        "image_ch":1,
        "image_ch": 1,
        "number_of_generated_batches": 8,
        "clamp_val": 1.8
    },
}
