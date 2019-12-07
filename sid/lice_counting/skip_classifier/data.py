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

PATH = '/data/sid/lice_counting_anns/all_pens_since_october.pkl'
IMAGE_FIELD = 'left_crop_url'
LABEL_FIELD = 'state'
ALLOWED_LABELS = ['QA', 'SKIPPED_ANN'] 
DATA_PATH = 'data_dir' 
NUM_SAMPLES = 100000

def download_images_to_local_dir():
    print('Loading dataframe...')
    s3 = boto3.resource('s3')
    frame = pd.read_pickle(PATH)
    frame = frame[frame[LABEL_FIELD].isin(ALLOWED_LABELS) & frame[IMAGE_FIELD].notnull()]
    print(frame[LABEL_FIELD].unique())
    print('Building random sample...')
    frame = frame.sample(n=NUM_SAMPLES)
    image_out_path = os.path.join(DATA_PATH, 'images')

    for label in ALLOWED_LABELS:
        os.makedirs(os.path.join(DATA_PATH, label))

    times = []
    print(f'Downloading dataset of size:{len(frame)}...')

    for i, (_, row) in tqdm(enumerate(frame.iterrows())):
        start = time()
        image_url = row[IMAGE_FIELD]
        image_label = row[LABEL_FIELD]
        image_bucket, image_key = get_key(image_url)
        local_filename = os.path.join(DATA_PATH, image_label, (str(uuid4())))
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
    download_images_to_local_dir()
