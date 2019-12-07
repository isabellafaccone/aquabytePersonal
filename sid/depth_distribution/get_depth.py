import json
from functools import partial
import os
import tempfile
import boto3
import io


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from aquabyte.data_access_utils import RDSAccessUtils
from aquabyte.optics import pixel2world


class DepthReport:

    def __init__(self,
            pen_id,
            date_range,
            credentials=json.load(open(os.environ["PROD_SQL_CREDENTIALS"]))
        ) -> None:
        self.credentials = credentials
        self.pen_id
        self.date_range

    def load_input(self):
        s3 = RDSAccessUtils(credentials)
        query = f"""SELECT * FROM keypoint_annotations WHERE (pen_id = {self.pen_id}) AND (captured_at BETWEEN '{date_range[0]}' AND '{date_range[1]})')"""
        keypoint_table = s3.extract_from_database(query)
        return keypoint_table

    def clean_input(self, keypoint_table):
        required_keys = ['leftCrop', 'rightCrop']
        def check_keypoints(d):
            for k in required_keys:
                if k not in d:
                    return False
            return True


        keypoint_table = keypoint_table[keypoint_table.keypoints.notnull()]
        keypoint_table[keypoint_table['keypoints'].apply(check_keypoints)]
        return keypoint_table


    def get_keypoints(self, keypoint_table):

        def get_row_dist(row):
            kps = pixel2world(
                row['keypoints']['leftCrop'],
                row['keypoints']['rightCrop'],
                row['camera_metadata']
            )
            ys = [coord[1] for coord in kps.values()]
            return np.median(ys)

        keypoint_table['distance'] = keypoint_table.apply(get_row_dist), axis=1)
        return keypoint_table

    def build_report(self, keypoint_table, save_path):
        fig, axes = plt.subplots(ncols=2)

        # Depth distribution
        keypoint_table['distance'].plot.hist(bins=20, ax=axes[0])
        mean = keypoint_table['distance'].mean()
        median = keypoint_table['distance'].median()}
        axes[0].set_title("Depth Distribution, mean={mean}, median={median}")

        # Depth over time
        keypoint_table['day_of_year'] = keypoint_table.captured_at.apply(lambda x: x.dayofyear)
        keypoint_table.groupby('day_of_year')['distance'].aggregate('mean').plot(ax=axes[1])
        axes[1].set_title('Depth over time')

        fig.save_fig(save_path)

    def get_bucket_key(url):
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

    def get_image(bucket, key):
        s3 = boto3.resource('s3')
        tmp = tempfile.NamedTemporaryFile(delete=False)
        bucket = s3.Bucket(bucket)
        image_object = bucket.Object(key)
        image_bytes = io.BytesIO(image_object.get()['Body'].read())
        img = mpimg.imread(image_bytes, 'jp2')
        return img

    def plot_images_within_depth(keypoint_table, depth_range, savepath, num_samples=5, step_size=0.1, ):

        num_images = num_samples*2
        fig, axes = plt.subplots(nrows=num_images, figsize=(10, num_images*5))

        current_image = 0
        print('Steps...')
        start, end = depth_range[0], depth_range[1]
        samples = keypoint_table[keypoint_table['distance'].between(start, end)].sample(num_samples)
        for i, (_,  row) in enumerate(samples.iterrows()):
            for side_url in ['left_image_url', 'right_image_url']:
                url = row[side_url]

                ax = axes[current_image]
                current_image += 1

                bucket, key = get_bucket_key(url)
                img = get_image(bucket, key)

                ax.imshow(img)

                side = side_url.split('_')[0]

                side_lab = f"{side} image"
                sample = f"Sample: {i}"
                distance = f"Distance: {round(row['distance'], 2)}"
                ax.set_title(" ".join([side_lab, sample, distance]), rotation = 'horizontal')

                crop_metadata = row[f'{side}_crop_metadata']

                def clean_dict(d):
                    for k in d:
                        if isinstance(d[k], float):
                            d[k] = round(d[k], 2)
                    return d

                crop_metadata = clean_dict(crop_metadata)
                meta = "CROP METADATA\n" + str(crop_metadata)
                wrap_size =70
                meta = '\n'.join([meta[i: i+wrap_size] for i in range(0, len(meta), wrap_size)])
                ax.set_xlabel(meta)

                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()
        plt.savefig(savepath, dpi=1000)

    def plot_example_images(keypoint_table, save_dir, depth_range=np.arange(0.3, 1.5, 0.1)):
        depth_bounds = [(x[i], x[i+1]) for i, _ in enumerate(depth_range[:-1])]
        for depth_range in depth_bounds:
            savepath = os.path.join(save_dir, f'depth_{depth_range[0]}-{depth_range[1]}.jpg')
            plot_images_within_depth(keypoint_table, depth_range, savepath)

