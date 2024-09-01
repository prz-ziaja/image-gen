import json
from pathlib import Path
import pandas as pd
from image_gen.constants import DatasetRawKeys, DatasetTrainValKeys



def read_metadata(coco_path:str):
    coco_path = Path(coco_path)

    with open(coco_path / "raw" / "captions_val2017.json", 'r') as f:
        val_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    # with open(coco_path / "raw" / "captions_train2017.json", 'r') as f:
    #     train_captions = pd.DataFrame(json.load(f)[DatasetRawKeys.ANNOTATIONS])

    with open(coco_path / "validation" / "labels.json", 'r') as f:
        val_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    # with open(coco_path / "train" / "labels.json", 'r') as f:
    #     train_labels = pd.DataFrame(json.load(f)[DatasetTrainValKeys.IMAGES])

    # train = pd.merge(train_labels,train_captions,left_on='id', right_on='image_id')
    # train['is_train'] = True

    val = pd.merge(val_labels,val_captions,left_on='id', right_on='image_id')
    val['is_train'] = False

    # full = pd.concat([train, val], axis=0)

    # return full
    return val
