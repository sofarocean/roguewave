import hashlib
from roguewave.wavewatch3 import clone_restart_file, RestartFile, open_restart_file
from io import BytesIO
import os
import boto3

REMOTE_HASH = "811449d693c24cca014fbe212a9db011"
REMOTE_RESTART_FILE = "s3://sofar-wx-data-dev-os1/roguewave/restart001.ww3"
REMOTE_MOD_DEF = "s3://sofar-wx-data-dev-os1/roguewave/mod_def.ww3"
TEST_DIR = "."
LOCAL_FILE_NAME = "restart001.ww3"
REMOTE_WRITE_PATH = "s3://sofar-wx-data-dev-os1/roguewave/partial/"


def file_hash(file):
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def clone_remote() -> RestartFile:
    #
    local_file = os.path.join(TEST_DIR, LOCAL_FILE_NAME)
    if not os.path.exists(local_file):
        s3 = boto3.client("s3")
        s3.download_file(
            "sofar-wx-data-dev-os1", "roguewave/restart001.ww3", local_file
        )

    local_mod_file = os.path.join(TEST_DIR, "mod_def.ww3")
    if not os.path.exists(local_mod_file):
        s3 = boto3.client("s3")
        s3.download_file(
            "sofar-wx-data-dev-os1", "roguewave/mod_def.ww3", local_mod_file
        )

    return open_restart_file(local_file, local_mod_file)


def bytes_hash(_bytes):
    with BytesIO(_bytes) as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()
