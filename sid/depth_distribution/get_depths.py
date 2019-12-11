import json
import io
import boto3
import os

import numpy as np
import pandas as pd

from utils import get_bucket_key, get_stream

from aquabyte.data_access_utils import RDSAccessUtils
from aquabyte.optics import pixel2world

SQL_CREDENTIALS = json.load(open(os.environ["PROD_SQL_CREDENTIALS"]))
DEPTH_COL = 'depth_estimate'

class DepthsBase:

     REQUIRED_COLS = [DEPTH_COL, 'captured_at', 'left_image_url', 'right_image_url', 'left_crop_metadata']

     def check_depth_table(self, depth_table: pd.DataFrame) -> None:
         for col in self.REQUIRED_COLS:
             assert col in depth_table.columns        
             assert depth_table[col].isnull().sum() == 0, f'Found null vals for {col}'

     def get_depth_estimate(self):
         """Get a dataframe with fish distances and metadata

         TODO (@siddsach): Add more detailed docs here"""
         assert self.depth_table is not None
         return self.depth_table

class KeypointDepths(DepthsBase):

    def __init__(self,
            keypoint_table: pd.DataFrame=None
        ) -> None:
        self.keypoint_table = keypoint_table

    @classmethod
    def from_pen_id_and_date_range_using_db(cls, pen_id, date_range):
        s3 = RDSAccessUtils(SQL_CREDENTIALS)
        query = f"""SELECT * FROM keypoint_annotations WHERE (pen_id = {pen_id}) AND (captured_at BETWEEN '{date_range[0]}' AND '{date_range[1]})')"""
        keypoint_table = s3.extract_from_database(query)
        return cls(keypoint_table)

    def clean_input(self) -> None:
        """Filter keypoints to consider non-null inputs with all of needed keys"""
        required_keys = ['leftCrop', 'rightCrop']
        def check_keypoints(d):
            for k in required_keys:
                if k not in d:
                    return False
            return True


        self.keypoint_table = self.keypoint_table[self.keypoint_table.keypoints.notnull()]
        self.keypoint_table = self.keypoint_table[self.keypoint_table['keypoints'].apply(check_keypoints)]

    def get_depth_estimate(self):
        """Add depth_estimate column to df based on keypoints"""
        def get_row_dist(row):
            kps = pixel2world(
                row['keypoints']['leftCrop'],
                row['keypoints']['rightCrop'],
                row['camera_metadata']
            )
            ys = [coord[1] for coord in kps.values()]
            return np.median(ys)

        self.keypoint_table[DEPTH_COL] = self.keypoint_table.apply(get_row_dist, axis=1)

    def get_depth_table(self) -> pd.DataFrame:
        self.clean_input()
        self.get_depth_estimate()
        self.check_depth_table(self.keypoint_table)
        self.depth_table = self.keypoint_table

class TemplateMatchingDepths(DepthsBase):

    TEMPLATE_MATCHING_PARQUET_PREFIX = "s3://aquabyte-research/focus-distance/weekly/"

    def __init__(self, 
            parquet_table: pd.DataFrame,
            pen_id: int
        ):
        self.parquet_table = parquet_table[parquet_table.pen_id == pen_id]

    @classmethod
    def from_week_date_and_pen_id(cls, week_date: str, pen_id: int):
        bucket, key = get_bucket_key(cls.TEMPLATE_MATCHING_PARQUET_PREFIX)
        client = boto3.client('s3')
        paginator = client.get_paginator('list_objects')
        result = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=key)
        week_dates = [prefix.get('Prefix').split('/')[-2] for prefix in result.search('CommonPrefixes')]
        msg = f'Date: {week_date} not found in available dates:\n {week_dates}'
        assert week_date in week_dates, msg
        parquet_folder = key + week_date + "/"
        s3 = boto3.resource('s3')
        b = s3.Bucket(bucket)
        parquet_paths = [obj.key for obj in b.objects.filter(Prefix=parquet_folder) if obj.key.endswith('.parquet')]
        assert len(parquet_paths) == 1, f'should only be 1 parque file per week, found\n{parquet_paths}'
        parquet_stream = get_stream(bucket, parquet_paths[0])
        string_io = io.StringIO(parquet_stream.read())
        df = pd.read_parquet(string_io)
        return cls(df, pen_id)
        

    def clean_parquet(self):
        pass 

    def get_depth_table(self):
        self.check_table(self.parquet_table)
        self.depth_table = self.parquet_table
