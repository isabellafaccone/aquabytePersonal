import json
import os
import cv2
from research_lib.utils.data_access_utils import S3AccessUtils
from research.utils.data_access_utils import RDSAccessUtils
import uuid
from rectification import rectify
from sqlalchemy import MetaData


s3 = S3AccessUtils('/root/data')


def get_capture_keys(s3_bucket, s3_prefix):
    suffixes = ['frame.jpg']
    keygen = s3.get_matching_s3_keys(s3_bucket, s3_prefix, suffixes=['capture.json'])
    keys = []
    for key in keygen:
        keys.append(key)

    return keys


def get_image_pair_from_capture_key(capture_key):
    left_key = capture_key.replace('capture.json', 'left_frame.jpg')
    right_key = capture_key.replace('capture.json', 'right_frame.jpg')
    return (left_key, right_key)


def download_from_s3_url(s3_url):
    url_components = s3_url.replace('s3://', '').split('/')
    bucket = url_components[0]
    key = os.path.join(*url_components[1:])
    f = s3.download_from_s3(bucket, key)
    return f, bucket, key


def rectify_and_upload_images(s3_bucket, s3_prefix, image_pairs, stereo_parameters_url=None):

    left_urls, right_urls = [], []
    count = 0

    for left_key, right_key in image_pairs:
        
        # get unrectified full resolution frames
        left_full_res_frame_s3_url, right_full_res_frame_s3_url = [os.path.join('s3://', s3_bucket, key) for key in (left_key, right_key)]
        
        if stereo_parameters_url:
            left_full_res_frame_f, _, left_full_res_frame_key = download_from_s3_url(left_full_res_frame_s3_url)
            right_full_res_frame_f, _, right_full_res_frame_key = download_from_s3_url(right_full_res_frame_s3_url)
            stereo_parameters_f, _, _ = s3.download_from_url(stereo_parameters_url)
            
            # rectify into full resolution stereo frame pair and save to disk
            left_image_rectified, right_image_rectified = rectify(left_full_res_frame_f, right_full_res_frame_f, stereo_parameters_f)
            left_image_rectified_f = os.path.join(os.path.dirname(left_full_res_frame_f), 'left_frame.rectified.jpg')
            right_image_rectified_f = os.path.join(os.path.dirname(right_full_res_frame_f), 'right_frame.rectified.jpg')
            cv2.imwrite(left_image_rectified_f, left_image_rectified)
            cv2.imwrite(right_image_rectified_f, right_image_rectified)
            
            # upload rectified stereo frame pairs to s3
            left_rectified_full_res_frame_key = left_full_res_frame_key.replace('.jpg', '.rectified.jpg')
            right_rectified_full_res_frame_key = right_full_res_frame_key.replace('.jpg', '.rectified.jpg')
            s3.s3_client.upload_file(left_image_rectified_f, s3_bucket, left_rectified_full_res_frame_key)
            s3.s3_client.upload_file(right_image_rectified_f, s3_bucket, right_rectified_full_res_frame_key)
        
            left_image_rectified_s3_url = os.path.join('s3://', s3_bucket, left_rectified_full_res_frame_key)
            right_image_rectified_s3_url = os.path.join('s3://', s3_bucket, right_rectified_full_res_frame_key)
            left_urls.append(left_image_rectified_s3_url)
            right_urls.append(right_image_rectified_s3_url)

        else:
            left_urls.append(left_full_res_frame_s3_url)
            right_urls.append(right_full_res_frame_s3_url)
        
        
        print(count)
        count += 1
    
    image_url_pairs = list(zip(left_urls, right_urls))
    return image_url_pairs


def process_into_plali_records(image_url_pairs, metadata, workflow_id):

    values_to_insert = []
    for idx, image_url_pair in enumerate(image_url_pairs):
        id = str(uuid.uuid4())
        images = set(image_url_pair)
        priority = float(idx) / len(image_url_pairs)

        values = {
            'id': id,
            'workflow_id': workflow_id,
            'images': images,
            'metadata': metadata,
            'priority': priority
        }

        values_to_insert.append(values)

    return values_to_insert


def establish_plali_connection():
    rds = RDSAccessUtils(json.load(open(os.environ['PLALI_SQL_CREDENTIALS'])))
    engine = rds.sql_engine
    sql_metadata = MetaData()
    sql_metadata.reflect(bind=engine)
    return engine, sql_metadata


def insert_into_plali(values_to_insert, engine, sql_metadata):
    table = sql_metadata.tables['plali_images']
    conn = engine.connect()
    trans = conn.begin()
    conn.execute(table.insert(), values_to_insert)
    trans.commit()



def main():
    s3_bucket = 'aquabyte-images-raw'
    s3_prefix = 'environment=production/site-id=55/pen-id=97/date=2021-03-17/hour=10'
    stereo_parameters_url = 'https://aquabyte-stereo-parameters.s3-eu-west-1.amazonaws.com/L40029797_R40020184/2021-02-25T11:30:42.149694000Z_L40029797_R40020184_stereo-parameters.json'
    metadata = { 'type': 'Dale P3 pre-swap -- Round 2' }
    workflow_id = '00000000-0000-0000-0000-000000000055'

    capture_keys = get_capture_keys(s3_bucket, s3_prefix)
    image_pairs = [get_image_pair_from_capture_key(capture_key) for capture_key in capture_keys]
    image_url_pairs = rectify_and_upload_images(s3_bucket, s3_prefix)
    values_to_insert = process_into_plali_records(image_url_pairs, metadata, workflow_id)
    engine, sql_metadata = establish_plali_connection()
    insert_into_plali(values_to_insert[:1], engine, sql_metadata)
