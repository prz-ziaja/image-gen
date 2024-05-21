import s3fs

from image_gen.constants.secret import s3_secrets


def get_s3_fs():
    fs = s3fs.S3FileSystem(anon=False, **s3_secrets)
    return fs
