import os
import boto3
from uuid import uuid4
import json
from time import time

import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Assume we have a CSV with annotations

SAMPLED_DATA_DIR = '/root/data/sid/skip_classifier_datasets/sampled_datasets/'
SAMPLED_DATA_FNAME = 'qa_accept_cogito_skips_03-04-2020'
IMAGE_FIELD = 'left_crop_url'
LABEL_FIELD = 'label'
ALLOWED_LABELS = ['SKIP', 'ACCEPT']
MODEL_DATA_PATH = '/root/data/sid/skip_classifier_datasets/model_datasets/'
NUM_SAMPLES = 100000

def download_images_to_local_dir(fname=SAMPLED_DATA_FNAME):
    print('Loading dataframe...')
    s3 = boto3.resource('s3')
    frame = pd.read_csv(os.path.join(SAMPLED_DATA_DIR, fname+ '.csv'))
    image_out_path = os.path.join(MODEL_DATA_PATH, fname, 'images')

    for label in ALLOWED_LABELS:
        path = os.path.join(image_out_path, label)
        os.makedirs(path, exist_ok=True)

    times = []
    print(f'Downloading dataset of size:{len(frame)}...')

    for i, (_, row) in tqdm(enumerate(frame.iterrows())):
        start = time()
        image_url = row[IMAGE_FIELD]
        image_label = row[LABEL_FIELD]
        image_bucket, image_key = get_key(image_url)
        local_filename = os.path.join(image_out_path, image_label, (str(uuid4())))
        s3.meta.client.download_file(image_bucket, image_key,  local_filename + '_crop.jpg')
        # Save metadata in case we need it
        row.to_json(local_filename + '_metadata.json')
        times.append(time()-start)

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
    useful_labels = [
            'BLURRY',
            'BAD_CROP',
            'BAD_ORIENTATION',
            'OBSTRUCTION',
            'TOO_DARK'
    ]
    for lab in useful_labels:
        fname = f'qa_accept_{lab}_skips_03-04-2020'
        download_images_to_local_dir(fname=fname)
    download_images_to_local_dir('qa_accept_cogito_skips_03-04-2020_100k')
