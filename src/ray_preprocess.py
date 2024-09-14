import importlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray

import utils as u


def main(dataset_module_name):
    dataset_module = importlib.import_module(dataset_module_name)
    source_loader_module, source_loader_name, source_loader_args = (
        dataset_module.source_loader
    )
    ray_source_connector_module, ray_source_connector_name, ray_source_connector_args = (
        dataset_module.ray_source_connector
    )
    _, output_writer_name, writer_args = (
        dataset_module.output_writer
    )

    print(source_loader_module, source_loader_name)
    source_loader = importlib.import_module(source_loader_module).__getattribute__(
        source_loader_name
    )

    ray_source_connector = importlib.import_module(ray_source_connector_module).__getattribute__(ray_source_connector_name)

    source_raw_data = source_loader(**source_loader_args)
    dataset = ray_source_connector(source_raw_data, **ray_source_connector_args)

    for plugin_module, plugin_name, plugin_args, plugin_constructor_kwargs, map_args in dataset_module.plugins:
        plugin = importlib.import_module(plugin_module).__getattribute__(plugin_name)

        if "map_batches" in plugin.__name__:
            dataset = dataset.map_batches(plugin, fn_kwargs=plugin_args, fn_constructor_kwargs=plugin_constructor_kwargs, **map_args)
        elif "flat_map" in plugin.__name__:
            dataset = dataset.flat_map(plugin, fn_kwargs=plugin_args, fn_constructor_kwargs=plugin_constructor_kwargs, **map_args)
        else:
            dataset = dataset.map(plugin, fn_kwargs=plugin_args, fn_constructor_kwargs=plugin_constructor_kwargs, **map_args)

    dataset.__getattribute__(output_writer_name)(**writer_args)



if __name__ == "__main__":
    args = u.parse_args()
    u.ray_connect(args)
    main(args.pipeline_config)
