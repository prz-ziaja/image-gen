from ray import tune
import torch.nn as nn

IMAGE_SIZE = (32, 32)
DATASET_NAME = "image_gen.io.datasets.fmnist_001"

preprocessing = {
    "dataset_name": DATASET_NAME,
}

training = {
    "max_num_epochs": 1000,
    "max_num_samples": 1,
    "model_name": "image_gen.models.gnn_arch",
    "experiment_name": "FMNIST GNN SD",
    "scaling_config": {
        "num_workers": 1,
        #"resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.9},
        "use_gpu": True,
    },
    "data_module_hparams_shared": {
        "image_size": IMAGE_SIZE,
        "columns": ("image",),  # "encoded_sentence"),
        "batch_size": 64,
        "encoded_sentence_size": 10,
    },
    "data_module_kwargs": {
        "dataset_name": DATASET_NAME,
    },
    "hparams": {
        "lr": 0.001,  # tune.qloguniform(1e-4, 1e-2, 1e-5),
        "T": 150,
        "image_ch": 1,
        "t_start": 2e-4,
        "t_end": 4e-2,
        "loss_function": tune.choice(
            [
                nn.MSELoss(),
            ]
        ),
    },
}
