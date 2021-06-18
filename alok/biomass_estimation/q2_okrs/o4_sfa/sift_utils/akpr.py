import json
import os
import cv2
import numpy as np
from research.utils.data_access_utils import S3AccessUtils
from research.weight_estimation.keypoint_utils.body_parts import core_body_parts as body_parts
from picture import Picture
from scipy.spatial import Delaunay
from itertools import compress

s3_access_utils = S3AccessUtils('/root/data', json.load(open(os.environ['AWS_CREDENTIALS'])))
sift = cv2.KAZE_create()


def in_hull(p, hull):
    hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def apply_convex_hull_filter(kp, des, canonical_kps, bbox):
    X_canon_kps = np.array(list(canonical_kps.values()))
    X_kp = np.array([x.pt for x in kp]).reshape(-1, 2) + np.array([bbox[0], bbox[1]])
    is_valid = in_hull(X_kp, X_canon_kps)
    kp = list(compress(kp, is_valid))
    des = des[is_valid]
    return kp, des


def get_homography_and_matches(left_image_arr, right_image_arr,
                               left_kps, right_kps,
                               left_bbox, right_bbox,
                               good_perc=0.7, min_match_count=10):

    kp1, des1 = sift.detectAndCompute(left_image_arr, None)
    kp2, des2 = sift.detectAndCompute(right_image_arr, None)
    try:
        if not (des1.any() and des2.any()):
            return None, kp1, kp2, None, [0]
    except AttributeError:
        print("None type for detectAndComputer descriptor")
        return None, kp1, kp2, None, [0]
    
    # apply convex hull filter
    kp1, des1 = apply_convex_hull_filter(kp1, des1, left_kps, left_bbox)
    kp2, des2 = apply_convex_hull_filter(kp2, des2, right_kps, right_bbox)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    H, matches_mask = np.eye(3), []
    good = []

    # check that matches list contains actual pairs
    if len(matches) > 0:
        if len(matches[0]) != 2:
            print('Aborting: matches list does not contain pairs')
            return H, kp1, kp2, good, matches_mask

    for m, n in matches:
        if m.distance < good_perc * n.distance:
            good.append(m)

    if len(good) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    return H, kp1, kp2, good, matches_mask


def get_bbox(image_arr, kp, width=300, height=150):
    x_min, y_min, x_max, y_max = int(kp[0] - width / 2.0), int(kp[1] - height / 2.0), \
            int(kp[0] + width / 2.0), int(kp[1] + height / 2.0)

    if x_min < 0:
        x_min = 0
        x_max = x_min + width
    elif x_max >= image_arr.shape[1]:
        x_max = image_arr.shape[1] - 1
        x_min = x_max - width
    if y_min < 0:
        y_min = 0
        y_max = y_min + height
    elif y_max >= image_arr.shape[0]:
        y_max = image_arr.shape[0] - 1
        y_min = y_max - height

    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox


def generate_refined_keypoints(ann, left_image_arr, right_image_arr):

    left_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['leftCrop']}
    right_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['rightCrop']}

    left_crop_metadata = {
        'x_coord': ann['leftCrop'][0]['xFrame'] - ann['leftCrop'][0]['xCrop'],
        'y_coord': ann['leftCrop'][0]['yFrame'] - ann['leftCrop'][0]['yCrop']
    }
    right_crop_metadata = {
        'x_coord': ann['rightCrop'][0]['xFrame'] - ann['rightCrop'][0]['xCrop'],
        'y_coord': ann['rightCrop'][0]['yFrame'] - ann['rightCrop'][0]['yCrop']
    }

    left_fish_picture = Picture(image_arr=left_image_arr)
    right_fish_picture = Picture(image_arr=right_image_arr)
    left_fish_picture.enhance(in_place=True)
    right_fish_picture.enhance(in_place=True)
    
    left_items, right_items = [], []
    for bp in body_parts:
        left_kp, right_kp = left_kps[bp], right_kps[bp]
        left_bbox = get_bbox(left_fish_picture.get_image_arr(), left_kp)
        right_bbox = get_bbox(right_fish_picture.get_image_arr(), right_kp)
        left_patch = left_fish_picture.generate_crop(*left_bbox)
        right_patch = right_fish_picture.generate_crop(*right_bbox)
        H, _, _, _, matches_mask = get_homography_and_matches(left_patch, right_patch, left_kps, 
                                                              right_kps, left_bbox, right_bbox)

        num_matches = sum(matches_mask)
        if H is not None:
            local_left_kp = [left_kp[0] - left_bbox[0], left_kps[bp][1] - left_bbox[1]]
            local_right_kp = cv2.perspectiveTransform(
                np.array([local_left_kp[0], local_left_kp[1]]).reshape(-1, 1, 2).astype(float), H).squeeze()
            right_kp = [local_right_kp[0] + right_bbox[0], local_right_kp[1] + right_bbox[1]]

        left_item = {
            'keypointType': bp,
            'xCrop': left_kp[0],
            'yCrop': left_kp[1],
            'xFrame': left_crop_metadata['x_coord'] + left_kp[0],
            'yFrame': left_crop_metadata['y_coord'] + left_kp[1]
        }
        right_item = {
            'keypointType': bp,
            'xCrop': right_kp[0],
            'yCrop': right_kp[1],
            'xFrame': right_crop_metadata['x_coord'] + right_kp[0],
            'yFrame': right_crop_metadata['y_coord'] + right_kp[1]
        }
        
        left_items.append(left_item)
        right_items.append(right_item)

    modified_ann = {
        'leftCrop': left_items,
        'rightCrop': right_items
    }

    return modified_ann
