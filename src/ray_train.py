import importlib

import lightning.pytorch as pl
import mlflow
import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import image_gen


def build_train_func(model_module, data_module, data_location, experiment_name):
    def train_func(config):
        dm = data_module(
            dataset_path=data_location,
            columns=config["columns"],
            batch_size=config["batch_size"],
        )
        model = model_module.Model(config)

        setup_mlflow(
           config,
           experiment_name=experiment_name,
           #ctx.get_experiment_name()   ctx.get_local_world_size()  ctx.get_node_rank()         ctx.get_trial_dir()         ctx.get_trial_name()        ctx.get_world_rank()
           #ctx.get_local_rank()        ctx.get_metadata()          ctx.get_storage()           ctx.get_trial_id()          ctx.get_trial_resources()   ctx.get_world_size()
           run_name=train.get_context().get_trial_name(),
           tracking_uri="http://127.0.0.1:5000",
           #tags={"trial_dir":train.get_context().get_trial_dir()}
        )
        mlflow.pytorch.autolog()

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    return train_func


def tune_hms_asha(ray_trainer, search_space, num_epochs, num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_neg_f1",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


def main(module_name):
    pipeline_config_module = importlib.import_module(
        f"image_gen.pipeline_configs.{module_name}"
    )
    model_name = pipeline_config_module.training["model_name"]
    dataset_name = pipeline_config_module.preprocessing["dataset_name"]
    model_hparams = pipeline_config_module.training["hparams"]
    scaling_config = pipeline_config_module.training["scaling_config"]

    model_module = importlib.import_module(f"image_gen.models.{model_name}")
    dataset_module = importlib.import_module(f"image_gen.io.datasets.{dataset_name}")

    data_location = dataset_module.output_path
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment(model_name)

    train_func = build_train_func(
        model_module, dataset_module.dataloader, data_location, experiment_name=model_name
    )

    # The maximum training epochs
    num_epochs = pipeline_config_module.training["max_num_epochs"]

    # Number of sampls from parameter space
    num_samples = pipeline_config_module.training["max_num_samples"]

    scaling_config = ScalingConfig(**scaling_config)

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_neg_f1",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="gloo"),
    )
    results = tune_hms_asha(
        ray_trainer, model_hparams, num_epochs, num_samples=num_samples
    )
    return results


if __name__ == "__main__":
    ray.init(runtime_env={"py_modules": [image_gen], "conda": "ray"})
    main("simple_cnn_arch")
