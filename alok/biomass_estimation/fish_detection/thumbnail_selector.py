import json
import os
import random
from research_lib.utils.data_access_utils import S3AccessUtils
from research_lib.utils.datetime_utils import get_dates_in_range

s3 = S3AccessUtils()
rds = RDSAccessUtils()

INBOUND_BUCKET = 'aquabyte-frames-resized-inbound'


def get_capture_keys(site_id, pen_id, start_date, end_date):
    """Take pen_id_time_range dataset as an input, and return list of paired_urls and corresponding
    crop_metadatas."""

    dates = get_dates_in_range(start_date, end_date)
    capture_keys = []
    for date in dates:
        s3_prefix = 'environment=production/site-id={}/pen-id={}/date={}'.format(site_id, pen_id,
                                                                                 date)

        generator = s3.get_matching_s3_keys(INBOUND_BUCKET, prefix=s3_prefix,
                                                         subsample=1.0,
                                                         suffixes=['capture.json'])

        these_capture_keys = [key for key in generator]
        capture_keys.extend(these_capture_keys)

    return capture_keys


def get_paired_urls_and_crop_metadatas(capture_keys):
    """Gets left urls, right urls, and crop metadatas corresponding to capture keys."""

    left_urls, crop_metadatas = [], []
    for capture_key in capture_keys:

        # get image URLs
        left_image_key = capture_key.replace('capture.json', 'left_frame.resize_512_512.jpg')
        left_image_url = os.path.join('s3://', INBOUND_BUCKET, left_image_key)
        left_urls.append(left_image_url)

        # get crop metadata
        crop_key = capture_key.replace('capture.json', 'crops.json')
        s3.download_from_s3(INBOUND_BUCKET, crop_key, custom_location='/root/data/crops.json')
        crop_metadata = json.load(open('/root/data/crops.json'))

        # TODO: process this crop metadata

        crop_metadatas.append(crop_metadata)

    return left_urls, crop_metadatas


def get_random_image_urls_and_crop_metadatas(site_id, pen_id, start_date, end_date):
    capture_keys = get_capture_keys(site_id, pen_id, start_date, end_date)
    capture_keys_subset = random.sample(capture_keys, 250)
    left_urls, crop_metadatas = get_paired_urls_and_crop_metadatas(capture_keys_subset)
    return left_urls, crop_metadatas


