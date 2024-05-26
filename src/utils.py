import argparse
from importlib import import_module

import ray


class ARGS:
    pipeline_module = "--pipeline-module"
    pipeline_config = "--pipeline-config"
    remote_host = "--remote-host"
    conda_env = "--conda-env"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        ARGS.pipeline_module,
        required=True,
        help="Module containing code to execute using ray. In this case mnist.",
    )
    parser.add_argument(
        ARGS.pipeline_config,
        required=True,
        help="Module contining configuration of pipeline.",
    )
    parser.add_argument(
        ARGS.remote_host,
        required=False,
        help="Address of remote ray head node. If not passed then ray will be run locally.",
    )
    parser.add_argument(
        ARGS.conda_env,
        required=False,
        default="ray",
        help="Name of the conda environment which should be utilized to run the code.",
    )
    parsed = parser.parse_args()

    return parsed


def function_builder(func, **kwargs):
    def inner(*inner_args, **inner_kwargs):
        return func(*inner_args, **kwargs, **inner_kwargs)

    inner.__name__ = func.__name__
    return inner


def ray_connect(args):
    pipeline_module = import_module(args.pipeline_module)

    ray.init(
        args.remote_host,
        runtime_env={"py_modules": [pipeline_module], "conda": args.conda_env},
    )


if __name__ == "__main__":
    args = parse_args()
    ray_connect(args)
