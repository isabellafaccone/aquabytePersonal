import json
import cv2
import numpy as np


def load_params(params_file):
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
    
    # perform rectification
    (R1, R2, P1, P2, Q, leftROI, rightROI) = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, 0)

    left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
    right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)
    
    return left_maps, right_maps


def remap_pair(left_img_path, right_img_path, left_maps, right_maps):
    
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    remap_left = cv2.remap(img_left, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    remap_right = cv2.remap(img_right, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return remap_left, remap_right


def rectify(left_image_f, right_image_f, stereo_parameters_f):
    left_maps, right_maps = load_params(stereo_parameters_f)
    left_image_rectified, right_image_rectified = remap_pair(left_image_f, right_image_f, left_maps, right_maps)
    return left_image_rectified, right_image_rectified

