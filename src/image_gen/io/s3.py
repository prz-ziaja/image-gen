import numpy as np
import ray

from image_gen.io.utils import get_s3_fs



def list_s3_dir(dir_path, fs, extension=""):
    return [
        file_path["name"]
        for file_path in fs.listdir(dir_path)
        if extension in file_path["name"]
    ]


def read_dir(dir_path):
    fs = get_s3_fs()
    files = list_s3_dir(dir_path, fs, ".npy")
    loaded_data = [
        np.load(fs.open(npy_file), allow_pickle=True).item() for npy_file in files
    ]
    return loaded_data


def read_s3_file____(row, fs):
    s3_file = row["item"][0]
    data = np.load(
        fs.open(
            s3_file,
            # "/ray/image_gen/raw/data_batch_1.npy"
        ),
        allow_pickle=True,
    ).item()

    data["data"] = data["data"].reshape([-1, 3, 32, 32]).astype(np.float32)
    data["labels"] = data["labels"].reshape([-1, 1])
    data["test"] = np.array(["test" in s3_file.lower()] * len(data["data"]))
    return data


def ray_read_s3_file_image_gen(row, fs):
    s3_file = row["item"][0]
    data = np.load(
        fs.open(
            s3_file,
            # "/ray/image_gen/raw/data_batch_1.npy"
        ),
        allow_pickle=True,
    ).item()

    data["data"] = data["data"].reshape([-1, 3, 32, 32]).astype(np.float32)
    data["labels"] = np.array(data["labels"], dtype=np.int32).reshape([-1, 1])
    data["test"] = np.array(["test" in s3_file.lower()] * len(data["data"]))
    return data


def ray_read_image_gen_raw(dir_path):
    fs = get_s3_fs()
    files = ray.data.from_items(list_s3_dir(dir_path, fs, ".npy"))
    dataset = files.map_batches(
        function_builder(ray_read_s3_file_image_gen, fs=fs), batch_size=1
    )
    return dataset


def ray_write_results(dataset, dir_path, columns):
    fs = get_s3_fs()
    dataset.write_numpy(dir_path, column=columns, filesystem=fs)
