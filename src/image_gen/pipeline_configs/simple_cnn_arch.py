from ray import tune
import torch.nn as nn
from image_gen.constants import plugin_spec, function_spec

IMAGE_SIZE = (128, 128)
DATASET_NAME = "image_gen.io.datasets.coca_001"

preprocessing = {
    "dataset_name": DATASET_NAME,
}

training = {
    "max_num_epochs": 4,
    "max_num_samples": 10,
    "model_name": "image_gen.models.simple_cnn_arch",
    "experiment_name": "generator",
    "scaling_config": {
        "num_workers": 1,
        #"resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.4},
        "use_gpu": True,
    },
    "dataloader_hparams_shared":{
        "image_size": IMAGE_SIZE,
        "columns": ("hsv", "labels"),
    },
    "dataloader_args":{
        "dataset_name": DATASET_NAME,
    },
    "hparams": {
        "layer_1_size": tune.qrandint(8, 24),
        "layer_2_size": tune.qrandint(32, 56),
        "lr": tune.qloguniform(1e-4, 1e-2, 1e-5),
        "kernel_size": 3,
        "stride": 2,
        "batch_size": 128,
        "loss_function": tune.choice([nn.CrossEntropyLoss(), nn.MultiMarginLoss()])
    },
}
