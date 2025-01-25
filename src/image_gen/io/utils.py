import s3fs
from pyarrow import fs

from image_gen.constants.secret import s3_secrets, s3_secrets_pa


def get_s3_fs_pa():
    fs_pa = fs.S3FileSystem(**s3_secrets_pa)
    return fs_pa


def get_s3_fs():
    fs = s3fs.S3FileSystem(anon=False, **s3_secrets)
    return fs
