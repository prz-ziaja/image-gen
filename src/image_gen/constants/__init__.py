from collections import namedtuple
from dataclasses import dataclass

function_spec = namedtuple(
    "function_spec", ("module_name", "function_name", "fn_kwargs")
)
plugin_spec = namedtuple(
    "plugin_spec",
    ("module_name", "plugin_name", "fn_kwargs", "fn_constructor_kwargs", "map_args"),
)

DATA_LOCATIONS = {"LOCAL_FS", "S3"}


@dataclass
class DatasetRawKeys:
    INFO = "info"
    LICENSE = "license"
    IMAGES = "images"
    ANNOTATIONS = "annotations"
    CATEGORIES = "categories"


@dataclass
class DatasetTrainValKeys:
    INFO = "info"
    LICENSE = "license"
    IMAGES = "images"
    ANNOTATIONS = "annotations"
