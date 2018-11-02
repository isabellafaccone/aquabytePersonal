import csv
import glob
import json
import os

import cv2
import numpy as np


def get_matching_s3_keys(s3_client, bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


class Rectification:
    def __init__(self, base_folder, experiments_list, rectification_files):
        self.base_folder = base_folder
        exp2cal = self._load_exp_calibration_mapping()
        mapdict = {}
        for recfile in rectification_files:
            left_maps, right_maps = self._load_params(recfile)
            exp = recfile.split('/')[-2]
            mapdict[exp] = {}
            mapdict[exp]['left'] = left_maps
            mapdict[exp]['right'] = right_maps
        paths = self._list_paths(experiments_list)
        print('Rectifying {} images now'.format(len(paths)))
        self._rectify(paths, exp2cal, mapdict)

    def _list_paths(self, experiments_list):
        all_image_path = []
        for exp in experiments_list:
            all_image_path += glob.glob(os.path.join(self.base_folder, exp) + '/*.jpg')
        return all_image_path

    def _load_params(self, params_file):
        params = json.load(open(params_file))
        cameraMatrix1 = np.array(params['CameraParameters1']['IntrinsicMatrix']).transpose()
        cameraMatrix2 = np.array(params['CameraParameters2']['IntrinsicMatrix']).transpose()

        distCoeffs1 = params['CameraParameters1']['RadialDistortion'][0:2] + \
                      params['CameraParameters1']['TangentialDistortion'] + \
                      [params['CameraParameters1']['RadialDistortion'][2]]
        distCoeffs1 = np.array(distCoeffs1)

        distCoeffs2 = params['CameraParameters2']['RadialDistortion'][0:2] + \
                      params['CameraParameters2']['TangentialDistortion'] + \
                      [params['CameraParameters2']['RadialDistortion'][2]]
        distCoeffs2 = np.array(distCoeffs2)

        R = np.array(params['RotationOfCamera2']).transpose()
        T = np.array(params['TranslationOfCamera2']).transpose()
        imageSize = (4096, 3000)
        # rectification
        (R1, R2, P1, P2, Q, leftROI, rightROI) = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                                   distCoeffs2, imageSize, R, T, None, None, None, None,
                                                                   None, cv2.CALIB_ZERO_DISPARITY, 0)

        left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
        right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)

        return left_maps, right_maps

    def _load_exp_calibration_mapping(self):
        exp2cal = {}
        with open('/root/data/small_pen_data_collection/calibration.csv', 'r') as f:
            cr = csv.reader(f)
            for (i, row) in enumerate(cr):
                if i == 0:
                    continue
                exp2cal[row[0]] = row[1]
        return exp2cal

    def _rectify(self, paths, exp2cal, mapdict):
        for img_path in paths:
            exp_number = img_path.split('/')[-2]
            calfile = exp2cal[exp_number]
            left_maps = mapdict[calfile]['left']
            right_maps = mapdict[calfile]['right']
            img = cv2.imread(img_path)
            new_path = img_path.replace(exp_number, exp_number + '_rectified')
            if 'left' in img_path:
                img_remap = cv2.remap(img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            else:
                img_remap = cv2.remap(img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
            if not os.path.isdir(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            if not os.path.isfile(new_path):
                cv2.imwrite(new_path, img_remap)


