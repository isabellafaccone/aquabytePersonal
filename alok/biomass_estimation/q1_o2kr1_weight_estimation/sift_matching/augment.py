from collections import defaultdict
from typing import Dict
import numpy as np
import pandas as pd
from weight_estimation.utils import get_left_right_keypoint_arrs, get_ann_from_keypoint_arrs, \
    convert_to_world_point_arr, CameraMetadata


def augment(df: pd.DataFrame, augmentation_config: Dict) -> pd.DataFrame:
    print('hello')

    counts, edges = np.histogram(df.weight, bins=np.arange(0, 10000, 1000))
    trial_values = (5.0 / (counts / np.max(counts))).astype(int)
    max_jitter_std = augmentation_config['max_jitter_std']
    min_depth = augmentation_config['min_depth']
    max_depth = augmentation_config['max_depth']

    augmented_data = defaultdict(list)
    for idx, row in df.iterrows():

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

        weight = row.weight
        trials = trial_values[min(int(weight / 1000), len(trial_values) - 1)]
        for _ in range(trials):
            ann = row.keypoints
            X_left, X_right = get_left_right_keypoint_arrs(ann)
            wkps = convert_to_world_point_arr(X_left, X_right, cm)
            original_depth = np.median(wkps[:, 1])

            depth = np.random.uniform(min_depth, max_depth)
            scaling_factor = float(original_depth) / depth
            jitter_std = np.random.uniform(0, max_jitter_std)

            # rescale
            X_left = X_left * scaling_factor
            X_right = X_right * scaling_factor

            # add jitter
            X_left[:, 0] += np.random.normal(0, jitter_std, X_left.shape[0])
            X_right[:, 0] += np.random.normal(0, jitter_std, X_right.shape[0])

            # reconstruct annotation
            ann = get_ann_from_keypoint_arrs(X_left, X_right)
            augmented_data['annotation'].append(ann)
            augmented_data['fish_id'].append(row.fish_id)
            augmented_data['weight'].append(row.weight)
            augmented_data['kf'].append(row.k_factor)
            augmented_data['camera_metadata'].append(row.camera_metadata)

    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df