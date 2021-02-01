import json
import os
import random
import uuid

import pandas as pd
from sqlalchemy import MetaData

from thumbnail_selector import get_random_image_urls_and_crop_metadatas
from research.utils.data_access_utils import RDSAccessUtils


def construct_fish_detection_dataset(pen_id_time_ranges):

    df_by_dataset = dict()
    for dataset in pen_id_time_ranges:
        site_id, pen_id, start_date, end_date = dataset['site_id'], dataset['pen_id'], \
                                                dataset['start_date'], dataset['end_date']
        dataset_name = 'pen_id_{}_{}_{}'.format(pen_id, start_date, end_date)

        # get 512x512 image URLs and crop metadatas corresponding to this dataset
        left_urls, crop_metadatas = \
            get_random_image_urls_and_crop_metadatas(site_id, pen_id, start_date, end_date)

        # get 512x512 image URLs and crop metadatas corresponding to high density periods
        hd_left_urls, hd_crop_metadatas = \
            get_high_density_image_urls_and_crop_metadatas(site_id, pen_id, start_date, end_date)

        # prepare data for LALI insertion
        left_image_urls, metadatas = [], []
        for left_image_url, crop_metadata in zip(left_urls, crop_metadatas):
            left_image_urls.append(left_image_url)
            metadatas.append(crop_metadata)

        df = pd.DataFrame({
            'url': left_image_urls,
            'metadata': metadatas
        })

        df_by_dataset[dataset_name] = df

    return df_by_dataset


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





