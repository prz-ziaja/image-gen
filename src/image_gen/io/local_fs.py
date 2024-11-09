import json
from pathlib import Path
import pandas as pd
import os
import numpy as np
from PIL import Image
from image_gen.constants import DatasetRawKeys, DatasetTrainValKeys
from image_gen.io.Reader import ReaderAbstract


def read_metadata(coco_path: str):
    coco_path = Path(coco_path)

    with open(coco_path / "raw" / "captions_val2017.json", "r") as f:
        val_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    with open(coco_path / "raw" / "captions_train2017.json", "r") as f:
        train_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    with open(coco_path / "validation" / "labels.json", "r") as f:
        val_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    with open(coco_path / "train" / "labels.json", "r") as f:
        train_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    train = pd.merge(train_labels, train_captions, left_on="id", right_on="image_id")
    train["is_train"] = True

    val = pd.merge(val_labels, val_captions, left_on="id", right_on="image_id")
    val["is_train"] = False

    full = pd.concat([train, val], axis=0)

    return full


class Reader(ReaderAbstract):
    def __init__(self):
        pass

    def read_metadata(self, dir_path):
        dir_path = Path(dir_path)
        npy_files = filter(lambda x: ".npy" == x[-4:], os.listdir(dir_path))
        output = dict()
        for npy_file in npy_files:
            temp = np.load(str(dir_path / npy_file), allow_pickle=True).item()
            for column in temp.keys():
                output_column = output.get(column)
                if output_column is None:
                    output[column] = temp[column]
                else:
                    output[column] = np.concatenate([output_column, temp[column]])

        return output

    def read_image(self, file_path: str) -> np.ndarray:
        im = np.array(Image.open(file_path))

        return im
