import json
import os

import cv2
import numpy as np


class Rectification:
    """ rectify image pairs"""

    def __init__(self, base_folder, stereo_params_file):
        self.rectified_dir = rectified_dir
        left_maps, right_maps = self._load_params(stereo_params_file)
        raw_folder = os.path.join(base_folder, "raw")
        rectified_folder = os.path.join(base_folder, "rectified")
        raw_images = glob.glob(raw_folder + "/*.jpg")
        self._rectify(raw_images, rectified_folder, left_maps, right_maps)


    def _load_params(self, stereo_params_file):
        """ load rectification parameters and create maps"""
        params = json.load(open(stereo_params_file))
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


    def _rectify(self, raw_image_files, rectified_dir, left_maps, right_maps):
        """ rectify pair of images """
        for raw_image_f in raw_image_files:
            img = cv2.imread(raw_image_f)
            rectified_image_f = os.path.join(rectified_dir, os.path.basename(raw_image_f))
            if 'left' in raw_image_f:
                rectified_image = cv2.remap(img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            else:
                rectified_image = cv2.remap(img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
            if not os.path.isdir(os.path.dirname(rectified_image_f)):
                os.makedirs(os.path.dirname(rectified_image_f))
            if not os.path.isfile(rectified_image_f):
                cv2.imwrite(rectified_image_f, rectified_image)
