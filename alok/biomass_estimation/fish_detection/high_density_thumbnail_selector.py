from collections import defaultdict
import json
import os
from urllib.parse import urlparse
import pandas as pd
from research_lib.utils.data_access_utils import S3AccessUtils, RDSAccessUtils

s3 = S3AccessUtils('/root/data')
rds = RDSAccessUtils()

INBOUND_BUCKET = 'aquabyte-frames-resized-inbound'


def get_dw_biomass_data(pen_id, start_date, end_date):
    query = """
        SELECT * 
        FROM prod.biomass_computations
        WHERE pen_id={0} 
        AND captured_at BETWEEN '{1}' and '{2}'
        AND date_part('hour', captured_at) >= {3} 
        AND date_part('hour', captured_at) <= {4}
    """.format(pen_id, start_date, end_date, 7, 15)

    df = rds.extract_from_database(query)
    return df


def _get_bucket_key(url):
    parsed_url = urlparse(url, allow_fragments=False)
    if parsed_url.netloc.startswith('s3'):
        url_components = parsed_url.path.lstrip('/').split('/')
        bucket, key = url_components[0], os.path.join(*url_components[1:])
    else:
        bucket = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')
    return bucket, key


def get_full_frame_resize_url(left_crop_url):
    _, key = _get_bucket_key(left_crop_url)
    key_components = key.split('/')[:-1]
    file_basename = 'left_frame.resize_512_512.jpg'
    full_frame_resize_s3_url = os.path.join('s3://', 'aquabyte-frames-resized-inbound', *key_components, file_basename)
    return full_frame_resize_s3_url


def get_full_frame_resize_dataset(df_dw):
    full_frame_to_crop_urls = defaultdict(list)
    for idx, row in df_dw.iterrows():
        left_crop_url = row.left_crop_url
        left_full_frame_resize_url = get_full_frame_resize_url(left_crop_url)
        full_frame_to_crop_urls[left_full_frame_resize_url].append(row.left_crop_url)        

    left_image_urls, counts = [], []
    for k, v in full_frame_to_crop_urls.items():
        left_image_urls.append(k)
        counts.append(len(v))

    df_resize = pd.DataFrame({
        'left_image_url': left_image_urls,
        'fish_count': counts
    })

    return df_resize


def get_high_density_image_urls_and_crop_metadatas(pen_id, start_date, end_date):
    print('Getting raw biomass computations from data warehouse...')
    df_dw = get_dw_biomass_data(pen_id, start_date, end_date)
    
    print('Done! Now getting dataframe consisting of corresponding resized thumbnails and crop counts')
    df_resize = get_full_frame_resize_dataset(df_dw)
    df_resize = df_resize.sort_values('fish_count', ascending=False).head(500)

    print('Done! Now extracting crop metadata information and returning image URLs and crop metadatas')
    left_urls, crop_metadatas = [], []
    for idx, row in df_resize.iterrows():

        left_url = row.left_image_url
        left_urls.append(left_url)

        _, left_image_key = _get_bucket_key(left_url)
        crop_key = os.path.join(*left_image_key.split('/')[:-1], 'crops.json')
        
        s3.download_from_s3(INBOUND_BUCKET, crop_key, custom_location='/root/data/crops.json')
        crop_metadata = json.load(open('/root/data/crops.json'))

        anns = crop_metadata['annotations']
        if anns:
            left_image_anns = [ann for ann in anns if ann['image_id'] == 1]
            crop_metadatas.append(left_image_anns)
        else:
            crop_metadatas.append([])

    return left_urls, crop_metadatas
