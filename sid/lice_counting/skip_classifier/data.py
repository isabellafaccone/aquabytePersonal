import pandas as pd
import torch.nn as nn
import os

# Assume we have a CSV with annotations

PATH = ''
IMAGE_FIELD = 'left_crop_url'
LABEL_FIELD = 'state'
DATA_PATH = '/data/sid/lice_count_skips/'


def download_images_to_local_dir():
    s3 = boto3.resource('s3')
    frame = pd.read_csv(PATH)
    image_out_path = os.path.join(DATA_PATH, 'images')
    for i, row in frame.iterrows():
        image_url = row[IMAGE_FIELD]
        image_bucket, image_key = get_key(image_url)
        local_filename = image_key.replace('/', '__SLASH__')
        s3.download_file(image_bucket, image_key, os.path.join(DATA_PATH, local_filename))
        print(local_filename)
        label = row[LABEL_FIELD]
        break

def get_key(url):
    # old style https://s3.amazonaws.com/<bucket>/<key>
    # new style https://<bucket>.s3.amazonaws.com/<key>
    splitted = url.split("/")
    # eg s3.amazonaws.com
    first_part = splitted[2].split('.')
    if len(first_part) != 3:
        # new style
        bucket = first_part[0]
        key = "/".join(splitted[3:])
    else:
        bucket=splitted[3]
        key = "/".join(splitted[4:])
    return bucket, key

if __name__ == '__main__':
    download_images_to_local_dir()
