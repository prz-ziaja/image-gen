from ray import tune
import torch.nn as nn
from image_gen.constants import plugin_spec, function_spec

IMAGE_SIZE = (64, 64)
DATASET_NAME = "image_gen.io.datasets.coca_001"

preprocessing = {
    "dataset_name": DATASET_NAME,
}

training = {
    "max_num_epochs": 10,
    "max_num_samples": 1,
    "model_name": "image_gen.models.simple_cnn_arch",
    "experiment_name": "Default",#"generator",
    "scaling_config": {
        "num_workers": 1,
        #"resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.4},
        "use_gpu": True,
    },
    "data_module_hparams_shared":{
        "image_size": IMAGE_SIZE,
        "columns": ("image", ),#"encoded_sentence"),
        "batch_size": 64,
    },
    "data_module_kwargs":{
        "dataset_name": DATASET_NAME,
        "reading_class": "image_gen.io.local_fs",
    },
    "hparams": {
        "lr": 0.0001,#tune.qloguniform(1e-4, 1e-2, 1e-5),
        "T": 60,
        "t_start": 6e-4,
        "t_end": 8e-2,
        "loss_function": tune.choice([nn.MSELoss(),])
    },
}