import importlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray
import torch

ray.init()


def read_spectrogram(config):
    spectrogram_id = list(config["spectrogram_id"].values())[0]
    spectrogram = pq.read_table(
        f"../../data/train_spectrograms/{spectrogram_id}.parquet"
    ).to_pandas()
    filled_spectrogram = spectrogram.fillna(0).values
    return {"config": config, "spectrogram": filled_spectrogram}


class TorchPredictor:
    def __init__(self):
        model_source = importlib.import_module(f"models.{self.model_name}")
        self.model = model_source.HMSClassifier(self.model_config)
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.cuda()

    def __call__(self, batch):
        inputs = torch.as_tensor(batch["processed"], dtype=torch.float32).cuda()
        with torch.inference_mode():
            predictions = torch.exp(self.model(inputs)).detach().cpu().numpy()
        return {
            "spec_id": batch["spec_id"],
            "seizure_vote": predictions[:, 0],
            "lpd_vote": predictions[:, 1],
            "gpd_vote": predictions[:, 2],
            "lrda_vote": predictions[:, 3],
            "grda_vote": predictions[:, 4],
            "other_vote": predictions[:, 5],
        }

    @classmethod
    def set_configuration(cls, config):
        cls.checkpoint_path = config["checkpoint_path"]
        cls.model_name = config["model_name"]
        cls.model_config = config["model_config"]


# def main(plugin_names):
TorchPredictor.set_configuration(
    {
        "checkpoint_path": "/home/przemek/ray_results/TorchTrainer_2024-03-09_11-32-44/TorchTrainer_5f0bd_00006_6_batch_size=32,layer_1_size=32,layer_2_size=64,lr=0.0006_2024-03-09_11-32-47/checkpoint_000015/checkpoint.ckpt",
        "model_name": "simple_cnn_arch",
        "model_config": {
            "lr": 0.001,
            "layer_1_size": 32,
            "layer_2_size": 64,
        },
    }
)
plugin_name = "001_log_scaling"

assignment = pd.read_csv("../../data/train.csv")
grouped = (
    assignment[
        [
            "spectrogram_id",
            "spectrogram_label_offset_seconds",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
    ]
    .groupby(
        [
            "spectrogram_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
    )
    .min("spectrogram_label_offset_seconds")
    .reset_index()
)
ids = grouped.spectrogram_id.unique()
configs = [
    grouped[grouped.spectrogram_id == spectrogram_id].to_dict()
    for spectrogram_id in ids
]
configs_set = ray.data.from_items([{"config": config} for config in configs])
plugin = importlib.import_module(f"preprocessing.{plugin_name}")
read_res = configs_set.map(lambda x: read_spectrogram(x["config"]))

processed = read_res.map(plugin.process_inference)
a = processed.take_batch()

predictions = processed.map_batches(
    TorchPredictor, batch_size=256, num_gpus=0.4, concurrency=2
)
predictions.write_csv("/tmp/hms_inference")


# if __name__ == "__main__":
#    main(["001_log_scaling"])
