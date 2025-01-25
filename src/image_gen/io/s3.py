import json
import pandas as pd
import numpy as np
from PIL import Image
from image_gen.constants import DatasetRawKeys, DatasetTrainValKeys
from image_gen.io.Reader import ReaderAbstract
from image_gen.io.utils import get_s3_fs


def read_metadata(coco_path: str):
    fs = get_s3_fs()

    print("open val captions")
    with fs.open(coco_path + "/raw/captions_val2017.json", "r") as f:
        val_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    print("open train captions")
    with fs.open(coco_path + "/raw/captions_train2017.json", "r") as f:
        train_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    print("open val labels")
    with fs.open(coco_path + "/validation/labels.json", "r") as f:
        val_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    print("open train labels")
    with fs.open(coco_path + "/train/labels.json", "r") as f:
        train_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    print("merge")
    train = pd.merge(train_labels, train_captions, left_on="id", right_on="image_id")
    train["is_train"] = True

    val = pd.merge(val_labels, val_captions, left_on="id", right_on="image_id")
    val["is_train"] = False

    full = pd.concat([train, val], axis=0)
    print("all done")

    return full


class Reader(ReaderAbstract):
    def __init__(self):
        self.fs = get_s3_fs()

    def read_metadata(self, dir_path):
        npy_files = filter(lambda x: ".npy" == x[-4:], self.fs.ls(dir_path))
        output = dict()
        for npy_file in npy_files:
            with self.fs.open(str(dir_path + "/" + npy_file)) as f:
                temp = np.load(f, allow_pickle=True).item()
            for column in temp.keys():
                output_column = output.get(column)
                if output_column is None:
                    output[column] = temp[column]
                else:
                    output[column] = np.concatenate([output_column, temp[column]])

        return output

    def read_image(self, file_path: str) -> np.ndarray:
        with self.fs.open(file_path) as f:
            im = np.array(Image.open(f))

        return im
