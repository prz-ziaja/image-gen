import importlib

import lightning.pytorch as pl
import mlflow
import torch
from ray import train, tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from lightning.pytorch.loggers import MLFlowLogger
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import utils as u


def build_train_func(model_module, data_module, data_module_args, experiment_name):
    def train_func(config):
        dm = data_module(**data_module_args)
        model = model_module.Model(config)

        smlf = setup_mlflow(
            config,
            experiment_name=experiment_name,
            # ctx.get_experiment_name()   ctx.get_local_world_size()  ctx.get_node_rank()         ctx.get_trial_dir()         ctx.get_trial_name()        ctx.get_world_rank()
            # ctx.get_local_rank()        ctx.get_metadata()          ctx.get_storage()           ctx.get_trial_id()          ctx.get_trial_resources()   ctx.get_world_size()
            run_name=train.get_context().get_trial_name(),
            tracking_uri="http://127.0.0.1:5000",
            create_experiment_if_not_exists=True,
            # tags={"trial_dir":train.get_context().get_trial_dir()}
        )

        mlflow.pytorch.autolog()
        mlf_logger = MLFlowLogger(
            run_id=smlf.active_run().info.run_id,
            experiment_name="Default",
            tracking_uri="http://127.0.0.1:5000",
        )
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=True,
            logger=mlf_logger,
        )
        torch.compile(model)
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    return train_func


def tune_hms_asha(ray_trainer, search_space, num_epochs, num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="train_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


def main(module_name):
    pipeline_config_module = importlib.import_module(module_name)
    model_name = pipeline_config_module.training["model_name"]
    dataset_name = pipeline_config_module.preprocessing["dataset_name"]
    scaling_config = pipeline_config_module.training["scaling_config"]
    experiment_name = pipeline_config_module.training["experiment_name"]
    data_module_args = (
        pipeline_config_module.training["data_module_hparams_shared"]
        | pipeline_config_module.training["data_module_kwargs"]
    )
    model_params = (
        pipeline_config_module.training["data_module_hparams_shared"]
        | pipeline_config_module.training["hparams"]
    )

    model_module = importlib.import_module(model_name)
    dataset_module = importlib.import_module(dataset_name)

    train_func = build_train_func(
        model_module,
        dataset_module.dataloader,
        data_module_args,
        experiment_name=experiment_name,
    )

    # The maximum training epochs
    num_epochs = pipeline_config_module.training["max_num_epochs"]

    # Number of sampls from parameter space
    num_samples = pipeline_config_module.training["max_num_samples"]

    scaling_config = ScalingConfig(**scaling_config)

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="train_loss",
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
        ray_trainer, model_params, num_epochs, num_samples=num_samples
    )
    return results


if __name__ == "__main__":
    args = u.parse_args()
    u.ray_connect(args)
    main(args.pipeline_config)
