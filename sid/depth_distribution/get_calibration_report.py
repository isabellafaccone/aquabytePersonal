import json
from functools import partial
import os
import boto3
import io
from typing import Tuple, Iterable, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from aquabyte.data_access_utils import RDSAccessUtils

from get_depths import DEPTH_COL
from utils import get_bucket_key, get_stream

DEPTH_REPORT_SAVENAME = 'depth_report.png'
IMAGE_DEPTH_DIR = 'example_images'

def get_image(stream):
    image_bytes = io.BytesIO(stream.read())
    img = mpimg.imread(image_bytes, 'jp2')
    return img
    

class CalibrationReport:
    """Report to help ops calibrate cameras

    produces plots, metrics, and images for calibrating the focusing depth_estimate
    of the camera"""

    def __init__(
            self,
            depth_table: pd.DataFrame,
            save_dir: str
        ) -> None:
        self.depth_table = depth_table 
        self.save_dir = save_dir

    def build_depth_report(self, depth_report_savename:str=DEPTH_REPORT_SAVENAME) -> Dict[str, float]:
        """Write depth calibration report to image

        Includes:
            * plot of depth distribution
            * calculation  mean, median, std to be returned as dict
            * plot of mean depth over time to see trend"""
        fig, axes = plt.subplots(ncols=2)

        # Depth distribution
        self.depth_table[DEPTH_COL].plot.hist(bins=20, ax=axes[0])
        mean = self.depth_table[DEPTH_COL].mean()
        median = self.depth_table[DEPTH_COL].median()
        std = self.depth_table[DEPTH_COL].std()
        metrics = {'mean': mean, 'median': median, 'std': std}
        title = "Depth Distribution, mean={mean}, median={median}"
        axes[0].set_title(title)

        if 'captured_at' in self.depth_table.columns:
            # Depth over time
            self.depth_table['day_of_year'] = self.depth_table.captured_at.apply(
                                                     lambda x: x.dayofyear)
            self.depth_table.groupby('day_of_year')[DEPTH_COL].aggregate('mean').plot(ax=axes[1])
            axes[1].set_title('Depth over time')
        else:
            axes[1].set_title('No captured_at variable, so we omit the depth over time plot')

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
        allowed_depth_estimates = self.depth_table[DEPTH_COL].between(start, end)
        keypoints_within_depth = self.depth_table[allowed_depth_estimates]
        samples = keypoints_within_depth.sample(num_samples)
        for i, (_, row) in enumerate(samples.iterrows()):
            for side_url in ['left_image_url', 'right_image_url']:
                url = row[side_url]

                ax = axes[current_image]
                current_image += 1

                bucket, key = get_bucket_key(url)
                stream = get_stream(bucket, key)
                img = get_image(stream)
 
                ax.imshow(img)

                side = side_url.split('_')[0]

                side_lab = f"{side} image"
                sample = f"Sample: {i}"
                depth_estimate = f"Distance: {round(row[DEPTH_COL], 2)}"
                ax.set_title(" ".join([side_lab, sample, depth_estimate]), rotation='horizontal')

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
        os.makedirs(os.path.join(self.save_dir, image_depth_dir), exist_ok=True)
        depth_bounds = [(depths[i], depths[i+1]) for i, _ in enumerate(depths[:-1])]
        for depth_range in depth_bounds:
            depth_savefile = f'depth_{depth_range[0]}-{depth_range[1]}.jpg'
            savepath = os.path.join(self.save_dir, image_depth_dir, depth_savefile)
            self.plot_images_within_depth(savepath, depth_range, **kwargs)

    def run(self):
        self.get_depth_estimate()
        self.build_depth_report()
        self.plot_example_images()

