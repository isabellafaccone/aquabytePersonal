import json
import os
import cv2
import numpy as np
from research.utils.data_access_utils import S3AccessUtils, RDSAccessUtils
from research.weight_estimation.keypoint_utils.body_parts import core_body_parts
from research.utils.image_utils import Picture
from scipy.spatial import Delaunay
from itertools import compress

def in_hull(p, hull):
    hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def apply_convex_hull_filter(kp, des, canonical_kps, bbox):
    X_canon_kps = np.array(list(canonical_kps.values()))
    X_kp = np.array([x.pt for x in kp]).reshape(-1, 2) + np.array([bbox['x_min'], bbox['y_min']])
    is_valid = in_hull(X_kp, X_canon_kps)
    kp = list(compress(kp, is_valid))
    des = des[is_valid]
    return kp, des


def get_homography_and_matches(sift, left_image, right_image,
                               left_kps, right_kps,
                               left_bbox, right_bbox,
                               good_perc=0.7, min_match_count=3):

    kp1, des1 = sift.detectAndCompute(left_image, None)
    kp2, des2 = sift.detectAndCompute(right_image, None)
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


def generate_sift_adjustment(bp, left_crop_metadata, left_fish_picture, left_kps, right_crop_metadata,
                             right_fish_picture, right_kps, sift):
    left_kp, right_kp = left_kps[bp], right_kps[bp]
    left_crop, left_bbox = left_fish_picture.generate_crop_given_center(left_kp[0], left_kp[1], 600, 200)
    right_crop, right_bbox = right_fish_picture.generate_crop_given_center(right_kp[0], right_kp[1], 600, 200)

    H, _, _, _, matches_mask = get_homography_and_matches(sift, left_crop, right_crop,
                                                          left_kps, right_kps,
                                                          left_bbox, right_bbox)
    num_matches = sum(matches_mask)
    if H is not None:
        local_left_kp = [left_kp[0] - left_bbox['x_min'], left_kp[1] - left_bbox['y_min']]
        local_right_kp = cv2.perspectiveTransform(
            np.array([local_left_kp[0], local_left_kp[1]]).reshape(-1, 1, 2).astype(float), H).squeeze()
        right_kp = [local_right_kp[0] + right_bbox['x_min'], local_right_kp[1] + right_bbox['y_min']]
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
    return left_item, right_item, num_matches


def generate_refined_keypoints(ann, left_crop_url, right_crop_url):

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

    left_fish_picture = Picture(image_url=left_crop_url)
    right_fish_picture = Picture(image_url=right_crop_url)
    left_fish_picture.enhance(in_place=True)
    right_fish_picture.enhance(in_place=True)
    sift = cv2.KAZE_create()
    left_items, right_items = [], []
    for bp in core_body_parts:
        left_item, right_item, num_matches = generate_sift_adjustment(bp, left_crop_metadata, left_fish_picture,
                                                                      left_kps, right_crop_metadata,
                                                                      right_fish_picture, right_kps, sift)
        left_items.append(left_item)
        right_items.append(right_item)
    modified_ann = {
        'leftCrop': left_items,
        'rightCrop': right_items
    }
    return modified_ann


def main():
    s3_access_utils = s3_access_utils = S3AccessUtils('/root/data', json.load(open(os.environ['AWS_CREDENTIALS'])))

    rds_access_utils = RDSAccessUtils(json.load(open(os.environ['PROD_SQL_CREDENTIALS'])))

    query = """
        SELECT * FROM keypoint_annotations
        WHERE pen_id=5
        AND captured_at BETWEEN '2019-06-05' AND '2019-07-02'
        AND keypoints is not null
        AND keypoints -> 'leftCrop' is not null
        AND keypoints -> 'rightCrop' is not null
        AND is_qa = FALSE
        LIMIT 1;
    """

    modified_anns = []

    df = rds_access_utils.extract_from_database(query)

    for idx, row in df.iterrows():
        # get annotation information
        ann = row.keypoints
        left_crop_url, right_crop_url = row.left_image_url, row.right_image_url
        left_crop_metadata, right_crop_metadata = row.left_crop_metadata, row.right_crop_metadata

        modified_ann = generate_refined_keypoints(ann, left_crop_url, right_crop_url)

        modified_anns.append(modified_ann)