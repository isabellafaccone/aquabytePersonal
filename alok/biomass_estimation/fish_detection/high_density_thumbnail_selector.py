from collections import defaultdict
import pandas as pd
from research_lib.utils.data_access_utils import RDSAccessUtils

rds = RDSAccessUtils()


def get_dw_biomass_data(pen_id, start_date, end_date):
    query = """
        SELECT * 
        FROM prod.biomass_computations
        WHERE pen_id={0} 
        AND captured_at BETWEEN {1} and {2}
        AND date_part('hour', captured_at) >= {3} 
        AND date_part('hour', captured_at) <= {4}
    """.format(pen_id, start_date, end_date, 7, 15)

    df = rds.extract_from_database(query)
    return df


def get_full_frame_512_url(left_crop_url):
    pass


def get_512_512_dataset(df):
    full_frame_to_crop_urls = defaultdict(list)
    for idx, row in df.iterrows():
        left_crop_url = row.left_crop_url
        left_full_frame_512_url = get_full_frame_512_url(left_crop_url)
        full_frame_to_crop_urls[left_full_frame_512_url].append(row.left_crop_url)
        full_frame_to_crop_urls[left_full_frame_512_url].append(row.right_crop_url)

    left_image_urls, right_image_urls, counts = [], [], []
    for k, v in full_frame_to_crop_urls.items():
        left_image_urls.append(k)
        right_image_urls.append(k.replace('left', 'right'))
        counts.append(len(v))

    df_512 = pd.DataFrame({
        'left_image_url': left_image_urls,
        'right_image_url': right_image_urls,
        'count': counts
    })

    return df_512


def get_high_density_image_urls_and_crop_metadatas(pen_id, start_date, end_date):
    df_dw = get_dw_biomass_data(pen_id, start_date, end_date)
    df_512 = get_512_512_dataset(df_dw)
    df_512 = df_512.sort_values('count', ascending=False)
    left_urls = list(df_512.left_image_url.head(500).values)
    right_urls = list(df_512.right_image_url.head(500).values)


