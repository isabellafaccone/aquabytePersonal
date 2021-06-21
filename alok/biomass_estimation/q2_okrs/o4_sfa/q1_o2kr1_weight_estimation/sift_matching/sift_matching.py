import cv2
import numpy as np
from research.utils.image_utils import Picture
from scipy.spatial import Delaunay
from itertools import compress


def in_hull(p, hull):
    hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def apply_convex_hull_filter(kp, des, canonical_kps):
    X_canon_kps = np.array(list(canonical_kps.values()))
    X_kp = np.array([x.pt for x in kp]).reshape(-1, 2)
    is_valid = in_hull(X_kp, X_canon_kps)
    kp = list(compress(kp, is_valid))
    des = des[is_valid]
    return kp, des


def get_homography_and_matches(sift, left_image, right_image,
                               left_kps, right_kps,
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
    kp1, des1 = apply_convex_hull_filter(kp1, des1, left_kps)
    kp2, des2 = apply_convex_hull_filter(kp2, des2, right_kps)

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


def generate_point_correspondences(left_crop_url, right_crop_url, ann):

    sift = cv2.KAZE_create()
    left_fish_picture = Picture(image_url=left_crop_url)
    right_fish_picture = Picture(image_url=right_crop_url)
    left_fish_picture.enhance(in_place=True)
    right_fish_picture.enhance(in_place=True)

    left_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['leftCrop']}
    right_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['rightCrop']}

    left_image = left_fish_picture.get_image_arr()
    right_image = right_fish_picture.get_image_arr()

    H, kp1, kp2, good, matches_mask = get_homography_and_matches(sift, left_image, right_image,
                                                                 left_kps, right_kps)

    left_corner_x = ann['leftCrop'][0]['xFrame'] - ann['leftCrop'][0]['xCrop']
    left_corner_y = ann['leftCrop'][0]['yFrame'] - ann['leftCrop'][0]['yCrop']
    right_corner_x = ann['rightCrop'][0]['xFrame'] - ann['rightCrop'][0]['xCrop']
    right_corner_y = ann['rightCrop'][0]['yFrame'] - ann['rightCrop'][0]['yCrop']
    left_corner = np.array([left_corner_x, left_corner_y])
    right_corner = np.array([right_corner_x, right_corner_y])

    
    left_points, right_points = [], []
    i = 0
    for m in good:
        if matches_mask[i] == 1:
            p1 = np.round(kp1[m.queryIdx].pt).astype(int) + left_corner
            p2 = np.round(kp2[m.trainIdx].pt).astype(int) + right_corner
            left_points.append(p1)
            right_points.append(p2)
        i += 1
    
    return np.array(left_points), np.array(right_points), left_corner, right_corner


# def generate_3d_point_cloud(left_points, right_points, cm):


