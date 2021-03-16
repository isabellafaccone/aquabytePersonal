import json
import os
from botocore.exceptions import ClientError

from research.utils.data_access_utils import RDSAccessUtils, S3AccessUtils

s3_access_utils = S3AccessUtils('/root/data', json.load(open(os.environ['AWS_CREDENTIALS'])))

from uuid import uuid4
from time import time
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from multiprocessing import Pool

# Assume we have a CSV with annotations
from config import SKIP_CLASSIFIER_DATASET_DIRECTORY, SKIP_CLASSIFIER_IMAGE_DIRECTORY

IMAGE_FIELD = 'url'
LABEL_FIELD = 'label'
ALLOWED_LABELS = ['SKIP', 'ACCEPT']

num_processes = 20

def download_frame(_row):
    _, row = _row

    start = time()
    image_url = row[IMAGE_FIELD]
    image_label = row[LABEL_FIELD]

    local_filename = os.path.join(row['image_out_path'], image_label, (str(uuid4())))

    try:
        local_f, _, _ = s3_access_utils.download_from_url(image_url)
        copyfile(local_f, local_filename + '_crop.jpg')
        # Save metadata in case we need i
        row.to_json(local_filename + '_metadata.json')

        return True
    except ClientError as e:
        print(e)
        return False

def download_images_to_local_dir(retraining_name, metadata):
    print('Loading dataframe...')

    image_out_path = os.path.join(SKIP_CLASSIFIER_IMAGE_DIRECTORY, retraining_name)
    dataset_file_name = os.path.join(SKIP_CLASSIFIER_DATASET_DIRECTORY, retraining_name + '.csv')

    frame = pd.read_csv(dataset_file_name)

    frame['image_out_path'] = image_out_path

    for label in frame['label'].unique():
        path = os.path.join(image_out_path, label)
        os.makedirs(path, exist_ok=True)

    total = len(frame)
    print(f'Downloading dataset of size:{total}...')
    
    pool = Pool(num_processes)
    
    results = list(tqdm(pool.imap(download_frame, frame.iterrows()), total=total))

    frame = frame[results]

    frame.to_csv(dataset_file_name)

    print('Number of skips', len(frame))

    print('Wrote file', dataset_file_name)

    metadata['num_rows'] = len(frame)

    return dataset_file_name, metadata
        
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
    dataset_file_name, metadata = download_images_to_local_dir('2021-03-16', {})

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
