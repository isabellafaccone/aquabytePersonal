from typing import Dict, List
import json
import os
import random
from research_lib.utils.data_access_utils import S3AccessUtils, RDSAccessUtils
from research_lib.utils.datetime_utils import get_dates_in_range

from botocore.exceptions import ClientError

s3 = S3AccessUtils('/root/data')
rds = RDSAccessUtils()

INBOUND_BUCKET = 'aquabyte-frames-resized-inbound'


def get_pen_site_mapping() -> Dict:
    query = 'select id, site_id from customer.pens;'
    pen_site_df = rds.extract_from_database(query)
    pen_site_mapping = dict(zip(pen_site_df.id.values, pen_site_df.site_id.values))
    return pen_site_mapping


PEN_SITE_MAPPING = get_pen_site_mapping()


def get_capture_keys(pen_id: int, start_date: str, end_date: str, inbound_bucket=INBOUND_BUCKET) -> List:
    """Take pen_id_time_range dataset as an input, and return list of paired_urls and corresponding
    crop_metadatas."""

    site_id = PEN_SITE_MAPPING[pen_id]
    dates = get_dates_in_range(start_date, end_date)
    capture_keys = []
    for date in dates:
        print('Getting capture keys for pen_id={}, date={}...'.format(pen_id, date))
        s3_prefix = 'environment=production/site-id={}/pen-id={}/date={}'.format(site_id, pen_id,
                                                                                 date)

        generator = s3.get_matching_s3_keys(inbound_bucket, prefix=s3_prefix,
                                                         subsample=1.0,
                                                         suffixes=['capture.json'])

        these_capture_keys = [key for key in generator]
        capture_keys.extend(these_capture_keys)

    return capture_keys


def get_image_urls_and_crop_metadatas(capture_keys):
    """Gets left urls, right urls, and crop metadatas corresponding to capture keys."""

    left_urls, crop_metadatas = [], []
    for capture_key in capture_keys:
        print(capture_key)

        # get image URLs
        left_image_key = capture_key.replace('capture.json', 'left_frame.resize_512_512.jpg')
        left_image_url = os.path.join('s3://', INBOUND_BUCKET, left_image_key)
        left_urls.append(left_image_url)

        # get crop metadata
        crop_key = capture_key.replace('capture.json', 'crops.json')

        try:
            s3.download_from_s3(INBOUND_BUCKET, crop_key, custom_location='/root/data/crops.json')
            crop_metadata = json.load(open('/root/data/crops.json'))

            anns = crop_metadata['annotations']
            if anns:
                left_image_anns = [ann for ann in anns if ann['image_id'] == 1]
                crop_metadatas.append(left_image_anns)
            else:
                crop_metadatas.append([])
        except ClientError as err:
            crop_metadatas.append([])

    return left_urls, crop_metadatas


def get_random_image_urls_and_crop_metadatas(pen_id, start_date, end_date):
    print('Getting all capture keys for pen between {} and {}...'.format(start_date, end_date))
    capture_keys = get_capture_keys(pen_id, start_date, end_date)

    print('Done! Subsetting 500 random data points and getting image URLs and crop metadatas')
    capture_keys_subset = random.sample(capture_keys, 500)
    left_urls, crop_metadatas = get_image_urls_and_crop_metadatas(capture_keys_subset)
    return left_urls, crop_metadatas