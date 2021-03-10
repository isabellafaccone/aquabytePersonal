import json
import os

aws_creds = json.load(open('/root/sid/credentials/sid_aws_credentials.json'))
os.environ['AWS_ACCESS_KEY_ID'] = aws_creds['aws_access_key_id']
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_creds['aws_secret_access_key']

import boto3
from uuid import uuid4
from time import time
import pandas as pd
from tqdm import tqdm

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

def download_images_to_local_dir(fname=SAMPLED_DATA_FNAME, downloaded=DOWNLOADED):
    print('Loading dataframe...')
    s3 = boto3.resource('s3')
    frame = pd.read_csv(os.path.join(SAMPLED_DATA_DIR, fname+ '.csv'))
    image_out_path = os.path.join(MODEL_DATA_PATH, fname, 'images')

    for label in frame['label'].unique():
        path = os.path.join(image_out_path, label)
        os.makedirs(path, exist_ok=True)

    times = []
    total = len(frame)
    print(f'Downloading dataset of size:{total}...')

    for i, (_, row) in tqdm(enumerate(frame.iterrows()), total=total):
        if downloaded is None or row[IMAGE_FIELD] not in downloaded:
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
