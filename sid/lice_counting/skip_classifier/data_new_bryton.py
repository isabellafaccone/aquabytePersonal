import json
import os

from research.utils.data_access_utils import RDSAccessUtils, S3AccessUtils

s3_access_utils = S3AccessUtils('/root/data', json.load(open(os.environ['AWS_CREDENTIALS'])))

import boto3
from uuid import uuid4
from time import time
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from multiprocessing import Pool

# Assume we have a CSV with annotations

SAMPLED_DATA_DIR = '/root/data/sid/needed_data/skip_classifier_datasets/sampled_datasets/'
SAMPLED_DATA_FNAME = '01152020_bodyparts'
IMAGE_FIELD = 'url'
LABEL_FIELD = 'label'
ALLOWED_LABELS = ['SKIP', 'ACCEPT']
MODEL_DATA_PATH = '/root/data/sid/needed_data/skip_classifier_datasets/images/'
DOWNLOADED = SAMPLED_DATA_FNAME + '_downloaded.json'
if os.path.exists(DOWNLOADED):
    print('partially downloaded...')
    DOWNLOADED = json.load(open(DOWNLOADED))

num_processes = 20
    
def download_frame(_row):
    _, row = _row
    
    if DOWNLOADED is None or row[IMAGE_FIELD] not in DOWNLOADED:
        start = time()
        image_url = row[IMAGE_FIELD]
        image_label = row[LABEL_FIELD]

        image_out_path = os.path.join(MODEL_DATA_PATH, SAMPLED_DATA_FNAME, 'images')
        local_filename = os.path.join(image_out_path, image_label, (str(uuid4())))

        local_f, _, _ = s3_access_utils.download_from_url(image_url)
        copyfile(local_f, local_filename + '_crop.jpg')

        # Save metadata in case we need it
        row.to_json(local_filename + '_metadata.json')
    
def download_images_to_local_dir():
    print('Loading dataframe...')
    
    frame = pd.read_csv(os.path.join(SAMPLED_DATA_DIR, SAMPLED_DATA_FNAME + '.csv'))
    
    image_out_path = os.path.join(MODEL_DATA_PATH, SAMPLED_DATA_FNAME, 'images')
    for label in frame['label'].unique():
        path = os.path.join(image_out_path, label)
        os.makedirs(path, exist_ok=True)

    total = len(frame)
    print(f'Downloading dataset of size:{total}...')
    
    pool = Pool(num_processes)
    
    list(tqdm(pool.imap(download_frame, frame.iterrows()), total=total))
        
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
    #useful_labels = [
    #        'BLURRY',
    #        'BAD_CROP',
    #        'BAD_ORIENTATION',
    #        'OBSTRUCTION',
    #        'TOO_DARK'
    #]
    #for lab in useful_labels:
    #    fname = f'qa_accept_{lab}_skips_03-04-2020'
    #    download_images_to_local_dir(fname=fname)
    #download_images_to_local_dir('qa_accept_cogito_skips_03-04-2020_100k')
