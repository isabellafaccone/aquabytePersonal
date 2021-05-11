from typing import Dict, List
import json, os
import numpy as np
import pandas as pd
from keras.models import load_model
from research.weight_estimation.akpd_utils.akpd_scorer import generate_confidence_score
from research.utils.data_access_utils import S3AccessUtils, RDSAccessUtils
from weight_estimation.utils import get_left_right_keypoint_arrs, convert_to_world_point_arr, \
    CameraMetadata


# generate raw GTSF dataframe from database
def generate_raw_df(start_date, end_date):
    rds = RDSAccessUtils(json.load(open(os.environ['PROD_RESEARCH_SQL_CREDENTIALS'])))
    query = """
        select * from research.fish_metadata a left join keypoint_annotations b
        on a.left_url = b.left_image_url 
        where b.keypoints -> 'leftCrop' is not null
        and b.keypoints -> 'rightCrop' is not null
        and b.captured_at between '{0}' and '{1}';
    """.format(start_date, end_date)
    df = rds.extract_from_database(query)
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.data.apply(lambda x: x['species'].lower()) == 'salmon'].copy(deep=True)
    qa_df = df[df.is_qa == True]
    cogito_df = df[(df.is_qa != True) & ~(df.left_image_url.isin(qa_df.left_image_url))]
    df = pd.concat([qa_df, cogito_df], axis=0)
    return df


def compute_akpd_score(akpd_scorer_network, keypoints: Dict, camera_metadata: Dict) -> float:
    input_sample = {
        'keypoints': keypoints,
        'cm': camera_metadata,
        'stereo_pair_id': 0,
        'single_point_inference': True
    }

    akpd_score = generate_confidence_score(input_sample, akpd_scorer_network)
    return akpd_score


def generate_akpd_scores(df: pd.DataFrame, akpd_scorer_f: str) -> List[float]:
    akpd_scorer_network = load_model(akpd_scorer_f)
    akpd_scores = []
    count = 0
    for _, row in df.iterrows():
        if count % 1000 == 0:
            print('Percentage complete: {}%'.format(round(100 * count / df.shape[0], 2)))
        count += 1
        akpd_score = compute_akpd_score(akpd_scorer_network, row.keypoints, row.camera_metadata)
        akpd_scores.append(akpd_score)
    return akpd_scores


def generate_depths(df: pd.DataFrame):
    depths = []
    for _, row in df.iterrows():
        annotation = row.keypoints
        camera_metadata = row.camera_metadata
        cm = CameraMetadata(
            focal_length=camera_metadata['focalLength'],
            focal_length_pixel=camera_metadata['focalLengthPixel'],
            baseline_m=camera_metadata['baseline'],
            pixel_count_width=camera_metadata['pixelCountWidth'],
            pixel_count_height=camera_metadata['pixelCountHeight'],
            image_sensor_width=camera_metadata['imageSensorWidth'],
            image_sensor_height=camera_metadata['imageSensorHeight']
        )
        X_left, X_right = get_left_right_keypoint_arrs(annotation)
        X_world = convert_to_world_point_arr(X_left, X_right, cm)
        depths.append(np.mean(X_world[:, ]))
    return depths


def prepare_gtsf_data(start_date: str, end_date: str, akpd_scorer_f: str,
                      akpd_score_cutoff: float, depth_cutoff: float) -> pd.DataFrame:
    df = generate_raw_df(start_date, end_date)
    print('Raw data loaded!')
    df = process(df)
    print('Data preprocessed!')
    df['k_factor'] = 1e5 * df.weight / df.data.apply(lambda x: x['lengthMms']**3).astype(float)
    df['akpd_score'] = generate_akpd_scores(df, akpd_scorer_f)
    df['depth'] = generate_depths(df)
    mask = (df.akpd_score > akpd_score_cutoff) & (df.depth < depth_cutoff)
    df = df[mask].copy(deep=True)
    return df


def main():
    akpd_scorer_url = 'https://aquabyte-models.s3-us-west-1.amazonaws.com/keypoint-detection-scorer/akpd_scorer_model_TF.h5'
    s3_access_utils = S3AccessUtils('/data', json.load(open(os.environ['AWS_CREDENTIALS'])))
    akpd_scorer_f, _, _ = s3_access_utils.download_from_url(akpd_scorer_url)
    df = prepare_gtsf_data('2019-06-01', '2019-06-10', akpd_scorer_f, 0.5, 1.0)
    print(df.shape)


if __name__ == '__main__':
    main()
