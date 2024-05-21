from ray import tune
import torch.nn as nn

preprocessing = {
    "dataset_name": "hsv_ds_00",
}

training = {
    "max_num_epochs": 4,
    "max_num_samples": 10,
    "model_name": "simple_cnn_arch",
    "scaling_config": {
        "num_workers": 1,
        #"resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.4},
        "use_gpu": True,
    },
    "hparams": {
        "columns": ("hsv", "labels"),
        "layer_1_size": tune.qrandint(8, 24),
        "layer_2_size": tune.qrandint(32, 56),
        "lr": tune.qloguniform(1e-4, 1e-2, 1e-5),
        "kernel_size": 3,
        "stride": 2,
        "batch_size": 128,
        "loss_function": tune.choice([nn.CrossEntropyLoss(), nn.MultiMarginLoss()])
    },
}
