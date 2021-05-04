from dataclasses import dataclass
import pickle
from typing import Any, List

import cv2
import numpy as np
from PIL import Image

MIN_MATCH_COUNT    = 10
GOOD_PERC          = 0.9
#sift               = cv2.xfeatures2d.SIFT_create()
#sift               = cv2.ORB_create()
sift               = cv2.AKAZE_create()
SCALE_PERCENT      = 100                                                   # percent
RANSAC_THRESH      = 10.0
INLIER_THRESH      = 30


#------------------------------------------------------------------------
# Data Serialization

@dataclass
class Keypoint:         # cv2.KeyPoint
    pt       : Any
    size     : Any
    angle    : Any
    response : Any
    octave   : Any
    class_id : Any


@dataclass
class KeypointsDescriptors:
    img_id      : str
    keypoints   : List[Keypoint]
    descriptors : Any # numpy array


def serialize_keypoints_descriptors(img_id, kp_des, file):
    kp, des = kp_des
    keypoints = [
        Keypoint(
            pt       = p.pt,
            size     = p.size,
            angle    = p.angle,
            response = p.response,
            octave   = p.octave,
            class_id = p.class_id,
        ) for p in kp
    ]
    item = KeypointsDescriptors(
        img_id      = img_id,
        keypoints   = keypoints,
        descriptors = des,
    )
    return pickle.dump(item, file)


def deserialize_keypoints_descriptors(file):
    item = pickle.load(file)
    keypoints = [cv2.KeyPoint(
        x         = kp.pt[0],
        y         = kp.pt[1],
        _size     = kp.size,
        _angle    = kp.angle,
        _response = kp.response,
        _octave   = kp.octave,
        _class_id = kp.class_id,
    ) for kp in item.keypoints]
    return item.img_id, (keypoints, item.descriptors)


##################################################
def enhance(image, clip_limit=5, sharp_grid_size=(21, 21), sharpen_weight=2.0):
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert image from LAB color model back to RGB color model
    enhanced_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    
    blurred = cv2.GaussianBlur(enhanced_image, sharp_grid_size, 0)
    enhanced_image = cv2.addWeighted(enhanced_image, sharpen_weight, blurred, 1-sharpen_weight, 0)
    
    #final_image = Image.fromarray(enhanced_image)
    
    return enhanced_image


def get_kp_desc(image):
    #width = int(image.shape[1] * SCALE_PERCENT / 100)
    #height = int(image.shape[0] * SCALE_PERCENT / 100)
    #dim = (width, height)
    #image_rz = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #img = enhance(image_rz)
    #kp1, des1 = sift.detectAndCompute(img, None)
    kp1, des1 = sift.detectAndCompute(image, None)
    return (kp1, des1)


def find_matches(kp_des1, kp_des2, is_sift=False):
    good = []
    (kp1, des1) = kp_des1
    (kp2, des2) = kp_des2
    
    matcher_type = cv2.DescriptorMatcher_BRUTEFORCE_SL2 if is_sift else cv2.DescriptorMatcher_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matches = matcher.knnMatch(des1, des2, 2)
    if (not matches) or len([m for m in matches if len(m) < 2]):
        return -1, False
    # print("Raw Matches %d" % len(matches))

    inliers = 0
    for m, n in matches:
        if m.distance < GOOD_PERC * n.distance:
            good.append(m)
    # print("Good Matches %d" % len(good))

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        for j in range(len(src_pts)):
            src_pts[j][0][0] *= 100/SCALE_PERCENT
            src_pts[j][0][1] *= 100/SCALE_PERCENT
        for j in range(len(dst_pts)):
            dst_pts[j][0][0] *= 100/SCALE_PERCENT
            dst_pts[j][0][1] *= 100/SCALE_PERCENT
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matches_mask = None

    #print("Final Inlier Matches %d" % inliers)
    return inliers, inliers > INLIER_THRESH
