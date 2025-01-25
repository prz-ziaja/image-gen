from typing import Any
import torchvision as tv
import PIL
from pathlib import Path
from image_gen.constants import DATA_LOCATIONS


class ImageLoader_map_batches:
    def __init__(self, filesystem_name, image_size, **kwargs):
        assert (
            filesystem_name in DATA_LOCATIONS
        ), f"{filesystem_name} not supported - pick one of {DATA_LOCATIONS}"

        self.to_tensor = tv.transforms.ToTensor()
        self.rescaler = tv.transforms.Rescale(image_size)

        if filesystem_name == "LOCAL_FS":
            self.open = open
        else:
            raise Exception("S3 not supported for now")

    def load_and_rescale_image(self, file_path):
        with self.open(file_path, "rb") as f:
            image = self.to_tensor(PIL.Image.open(f))

        rescaled = self.rescaler(image)
        return rescaled

    def __call__(
        self, batch, coco_path, filename_key, result_name, size, keep_source=False
    ) -> Any:
        filenames = batch.pop(filename_key)
        is_train = batch["is_train"]
        coco_path = Path(coco_path)

        images = list()
        for filename, training_sample in zip(filenames, is_train):
            if training_sample:
                file_path = coco_path / "train" / "data" / filename
            else:
                file_path = coco_path / "validation" / "data" / filename

            images.append(self.load_and_rescale_image(file_path))

        batch[result_name] = images

        if keep_source:
            batch[filename_key] = filenames

        return batch
