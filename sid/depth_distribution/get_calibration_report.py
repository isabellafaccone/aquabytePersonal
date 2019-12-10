import json
from functools import partial
import os
import tempfile
import boto3
import io
from typing import Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from aquabyte.data_access_utils import RDSAccessUtils
from aquabyte.optics import pixel2world

DEPTH_REPORT_SAVENAME = 'depth_report.png'
IMAGE_DEPTH_DIR = 'example_images'

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
        bucket = splitted[3]
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

class CalibrationReport:
    """Report to help ops calibrate cameras

    produces plots, metrics, and images for calibrating the focusing distance
    of the camera"""

    def __init__(
            self,
            keypoint_table: pd.DataFrame,
            save_dir: str
        ) -> None:
        self.keypoint_table = keypoint_table
        self.save_dir = save_dir

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

    def get_distance(self):
        """Add distance column to df based on keypoints"""
        def get_row_dist(row):
            kps = pixel2world(
                row['keypoints']['leftCrop'],
                row['keypoints']['rightCrop'],
                row['camera_metadata']
            )
            ys = [coord[1] for coord in kps.values()]
            return np.median(ys)

        self.keypoint_table['distance'] = self.keypoint_table.apply(get_row_dist, axis=1)

    def build_depth_report(self, depth_report_savename:str=DEPTH_REPORT_SAVENAME) -> Dict[str, float]:
        """Write depth calibration report to image

        Includes:
            * plot of depth distribution
            * calculation  mean, median, std to be returned as dict
            * plot of mean depth over time to see trend"""
        fig, axes = plt.subplots(ncols=2)

        # Depth distribution
        self.keypoint_table['distance'].plot.hist(bins=20, ax=axes[0])
        mean = self.keypoint_table['distance'].mean()
        median = self.keypoint_table['distance'].median()
        std = self.keypoint_table['distance'].std()
        metrics = {'mean': mean, 'median': median, 'std': std}
        title = "Depth Distribution, mean={mean}, median={median}"
        axes[0].set_title(title)

        # Depth over time
        self.keypoint_table['day_of_year'] = self.keypoint_table.captured_at.apply(
                                                 lambda x: x.dayofyear)
        self.keypoint_table.groupby('day_of_year')['distance'].aggregate('mean').plot(ax=axes[1])
        axes[1].set_title('Depth over time')

        report_savepath = os.path.join(self.save_dir, depth_report_savename)

        fig.savefig(report_savepath)
        return metrics


    def plot_images_within_depth(
            self,
            savepath: str,
            depth_range: Tuple[float, float],
            num_samples: int=5,
            dpi: int=1000
        ):
        """Plot example fish within a certain depth range

        Parameters
        ----------
        savepath: where to save the images for this depth
        depth_range: depth bounds to provide fish images for in meters
        num_samples: number of example images
        dpi: Dots per inches. Number for controlling resolution of output images"""

        num_images = num_samples*2
        fig, axes = plt.subplots(nrows=num_images, figsize=(10, num_images*5))

        current_image = 0
        start, end = depth_range[0], depth_range[1]
        print(type(start), type(end))
        allowed_distances = self.keypoint_table['distance'].between(start, end)
        keypoints_within_depth = self.keypoint_table[allowed_distances]
        samples = keypoints_within_depth.sample(num_samples)
        for i, (_, row) in enumerate(samples.iterrows()):
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
                ax.set_title(" ".join([side_lab, sample, distance]), rotation='horizontal')

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
        plt.savefig(savepath, dpi=dpi)

    def plot_example_images(self, 
            depths: Iterable=np.arange(0.3, 1.5, 0.1), 
            image_depth_dir: str=IMAGE_DEPTH_DIR, **kwargs
        ) -> None:
        """Plot example images for given depths
        
        depths: iterable of depth bin cutoffs in meters where we 
        find sample images within each bin
        image_depth_dir: directory for the images to be saved"""
        os.makedirs(os.path.join(self.save_dir, image_depth_dir))
        depth_bounds = [(depths[i], depths[i+1]) for i, _ in enumerate(depths[:-1])]
        for depth_range in depth_bounds:
            depth_savefile = f'depth_{depth_range[0]}-{depth_range[1]}.jpg'
            savepath = os.path.join(self.save_dir, image_depth_dir, depth_savefile)
            self.plot_images_within_depth(savepath, depth_range, **kwargs)

    def run(self):
        self.get_distance()
        self.build_depth_report()
        self.plot_example_images()

