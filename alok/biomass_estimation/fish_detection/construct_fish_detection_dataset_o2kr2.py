import json
import os
import random
import uuid

import pandas as pd
from sqlalchemy import MetaData

from thumbnail_selector import get_random_image_urls_and_crop_metadatas
from research.utils.data_access_utils import RDSAccessUtils


def construct_fish_detection_dataset(pen_id_time_range):

    pen_id, start_date, end_date = pen_id_time_range['pen_id'], pen_id_time_range['start_date'], pen_id_time_range['end_date']
    

    # get random 512x512 image URLs and crop metadatas corresponding to this dataset
    left_urls, crop_metadatas = get_random_image_urls_and_crop_metadatas(pen_id, start_date, end_date)

    # get 512x512 image URLs and crop metadatas corresponding to high density periods
    hd_left_urls, hd_crop_metadatas = get_high_density_image_urls_and_crop_metadatas(site_id, pen_id, start_date, end_date)

    # extend lists
    left_urls.extend(hd_left_urls)
    crop_metadatas.extend(hd_crop_metadatas)

    # prepare data for LALI insertion
    left_image_urls, metadatas = [], []
    data_spec_name = 'pen_id_{}_{}_{}'.format(pen_id, start_date, end_date)
    
    for left_image_url, crop_metadata in zip(left_urls, crop_metadatas):
        left_image_urls.append(left_image_url)
        metadata = { 'crops': crop_metadata }
        metadata['data_spec_name': data_spec_name]
        metadata.update(data_spec)

    df = pd.DataFrame({
        'url': left_image_urls,
        'metadata': metadatas
    })

    return df


def establish_plali_connection():
    rds = RDSAccessUtils(json.load(open(os.environ['PLALI_SQL_CREDENTIALS'])))
    engine = rds.sql_engine
    sql_metadata = MetaData()
    sql_metadata.reflect(bind=engine)
    return engine, sql_metadata


def process_into_plali_records(df):

    values_to_insert = []
    for idx, row in df.iterrows():
        id = uuid.uuid4()
        images = {row.url}
        metadata = row.metadata
        priority = random.random()

        values = {
            'id': id,
            'workflow_id': '00000000-0000-0000-0000-000000000050',
            'images': images,
            'metadata': metadata,
            'priority': priority
        }

        values_to_insert.append(values)

    return values_to_insert


def insert_into_plali(values_to_insert, engine, sql_metadata):
    table = sql_metadata.tables['plali_images']
    conn = engine.connect()
    trans = conn.begin()
    conn.execute(table.insert(), values_to_insert)
    trans.commit()


def main(pen_id_time_ranges):
    
    engine, sql_metadata = establish_plali_connection()
    for pen_id_time_range in pen_id_time_ranges:
        df = construct_fish_detection_dataset(pen_id_time_range)
        values_to_insert = process_into_plali_records(df)
        insert_into_plali(values_to_insert, engine, sql_metadata)
        




