IMAGE_SIZE = (32, 32)
DATASET_NAME = "image_gen.io.datasets.fmnist_001"

preprocessing = {
    "dataset_name": DATASET_NAME,
}

training = {
    "max_num_epochs": 1000,
    "max_num_samples": 1,
    "model_name": "image_gen.models.wgan_v1",
    "experiment_name": "Default",  # "generator",
    "scaling_config": {
        "num_workers": 1,
        # "resources_per_worker": {"CPU": 3},
        "resources_per_worker": {"GPU": 0.4},
        "use_gpu": True,
    },
    "data_module_hparams_shared": {
        "image_size": IMAGE_SIZE,
        "columns": ("image",),  # "encoded_sentence"),
        "batch_size": 128,
    },
    "data_module_kwargs": {
        "dataset_name": DATASET_NAME,
        "reading_class": "image_gen.io.local_fs",
    },
    "hparams": {
        "lr": 0.0001,  # tune.qloguniform(1e-4, 1e-2, 1e-5),
        "T": 40,
        "image_noise_t": 15,
        "noise_vector_len": 128,
        "image_ch": 1,
        "t_start": 1e-4,
        "t_end": 2e-2,
        "generated_image_per_step": 96,
    },
}
