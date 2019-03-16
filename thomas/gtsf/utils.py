import copy
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from skimage.measure import label
from sklearn.metrics import euclidean_distances

def load_image_annotations(annotation, new_shape, base_dir = "/root/data/gtsf_2.0/registration_test/"):
    """return the image with the mask"""
    image_path = os.path.join(base_dir, annotation['External ID'])
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    ratio_width = width / new_shape[0]
    ratio_height = height / new_shape[1]
    
    keypoints = {}
    for (label, value) in annotation["Label"].items():
        if label == "salmon":
            # mask
            polygon = [(k["x"], k["y"]) for k in value[0]["geometry"]]
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask = np.array(img)
            
            # reshape image
            image = cv2.resize(image, new_shape)
            mask = cv2.resize(mask, new_shape)
            masked_image = np.expand_dims(mask, axis=2)*image
        else:
            keypoints[label] = (int(value[0]["geometry"]["x"]/ratio_width), 
                                int(value[0]["geometry"]["y"]/ratio_height))
    
    keypoints_map = np.zeros_like(mask)
    delta=1
    for (kp, v) in keypoints.items():
        keypoints_map[v[1]-delta:v[1]+delta, v[0]-delta:v[0]+delta] = 1

    return masked_image, mask, keypoints, keypoints_map


def display_image_with_keypoints(image, keypoints, zoom=True):
    """plot image + keypoints"""
    keypoints_array = np.array([v for (kp, v) in keypoints.items()])
    buffer = 50
    if zoom:
        x, y = np.where(image[..., 0] > 0)
        xmin, xmax = np.min(x) - buffer, np.max(x) + buffer
        ymin, ymax = np.min(y) - buffer, np.max(y) + buffer
        
    plt.figure(figsize = (15, 10))
    plt.imshow(image[xmin: xmax, ymin: ymax, :])
    plt.scatter(keypoints_array[:, 0]-ymin, keypoints_array[:, 1]-xmin)
    plt.axis("off")
    plt.show()
    

def translate_moving(static_mask, moving_mask):
    """translate the moving image"""
    # centroids
    xs, ys = np.where(static_mask == 1)
    centroid_s = np.array((np.mean(xs), np.mean(ys)))
    xm, ym = np.where(moving_mask == 1)
    centroid_m = np.array((np.mean(xm), np.mean(ym)))
    
    # m to s
    translation_vector = centroid_s - centroid_m
    xmt, ymt = (xm+translation_vector[0], ym+translation_vector[1])
    
    moving_translated = np.zeros_like(static_mask)
    for (xi, yi) in zip(xmt,ymt):
        moving_translated[int(xi), int(yi)] = 1
        
    return moving_translated, translation_vector


def register(static, moving):
    """perform the image registration"""
    dim = static.ndim
    metric = SSDMetric(dim)    
    level_iters = [200, 100, 50, 25, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter = 50)
    mapping = sdr.optimize(static, moving)
    return mapping


def display_warped_keypoints(moving_image, warped_kp_map, translation_vector):
    """plot image + keypoints"""
    warped_keypoints = np.array(np.where(warped_kp_map>0))
    f, ax = plt.subplots(1,1, figsize=(20, 10))
    ax.imshow(moving_image)
    ax.scatter((warped_keypoints[1, :]-translation_vector[1]) , 
                  (warped_keypoints[0, :]-translation_vector[0]))
    ax.axis("off")
    plt.show()
    
    
def display_pairs_with_keypoints(moving_image, moving_kp_map, warped_kp_map, translation_vector):
    """ display ground truth + forward pass """
    moving_keypoints = np.array(np.where(moving_kp_map>0))
    warped_keypoints = np.array(np.where(warped_kp_map>0))
    
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(moving_image)
    ax[0].scatter(moving_keypoints[1, :], moving_keypoints[0, :])
    ax[0].axis("off")
    ax[1].imshow(moving_image)
    ax[1].scatter((warped_keypoints[1, :]-translation_vector[1]) , (warped_keypoints[0, :]-translation_vector[0]))
    ax[1].axis("off")
    plt.show()
    
def display_keypoints_gt_pred(moving_image, moving_kp_map, warped_kp_map, translation_vector):
    """ display ground truth + forward pass """
    moving_keypoints = np.array(np.where(moving_kp_map>0))
    warped_keypoints = np.array(np.where(warped_kp_map>0))
    
    f, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(moving_image)
    ax.scatter(moving_keypoints[1, :], moving_keypoints[0, :], c="r")
    ax.axis("off")
    ax.imshow(moving_image)
    ax.scatter((warped_keypoints[1, :]-translation_vector[1]) , (warped_keypoints[0, :]-translation_vector[0]), c="b")
    ax.axis("off")
    plt.show()
    
    
def calculate_errors(moving_keypoints, warped_kp_map, translation_vector):
    """ calculate errors between predicted keypoints and warped keypoints"""
    warped_keypoints = np.array(np.where(warped_kp_map>0)).transpose() # Nx2
    
    # get the cluster centers
    kmeans = KMeans(n_clusters=9, random_state=0).fit(warped_keypoints)
    centers = kmeans.cluster_centers_
    # get the predicted keypoints
    predicted_keypoints = []
    for i in range(centers.shape[0]):
         predicted_keypoints.append([int(centers[i, 1]-translation_vector[1]), int(centers[i,0]-translation_vector[0])])
    predicted_keypoints = np.array(predicted_keypoints)

    # ok now let's turn moving_keypoints into an array
    ground_truth_keypoints = []
    names = []
    for (k, v) in moving_keypoints.items():
        names.append(k)
        ground_truth_keypoints.append(v)
    ground_truth_keypoints = np.array(ground_truth_keypoints)
    
    # calculate the pairwise distances
    distance_matrix = pairwise_distances(ground_truth_keypoints, predicted_keypoints)
    
    # linear sum assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # calculate error
    for (i, (r, c)) in enumerate(zip(row_ind, col_ind)):
        print("Keypoint {}: prediction {} groundtruth {}".format(names[r], ground_truth_keypoints[r], predicted_keypoints[c]))
        print("Manhattan distance: {}".format(np.sum(np.abs(ground_truth_keypoints[r] - predicted_keypoints[c]))))
    
    
    
def rgb_to_hsv(r, g, b):
    """ rgb to hsv duh """
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v   
    
    
def create_mask(f, min_hsv, max_hsv):
    im = Image.open(f)
    im = im.convert('RGBA')
    im = im.resize((512, 512))
    original = copy.copy(im)
    # im = im.crop([50, 50, 100 ,100])
    pix = im.load()
    width, height = im.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = min_hsv
            max_h, max_s, max_v = max_hsv
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0, 0)
            else:
                continue
                # print(h,s,v)
    mask = np.asarray(im)[..., 3]
    mask.flags.writeable = True
    mask[mask > 0] = 1
    
    # let's erode
    mask = cv2.erode(mask, np.ones((3, 3)))
    
    # let's dilate
    mask = cv2.dilate(mask, np.ones((3, 3)))
    #mask = cv2.dilate(mask, np.ones((3, 3)))
    
    # let's get the fish
    labs = label(mask)
    center = np.array([[256, 256]])
    dist = 1e8
    # calculate labs centroids
    for l in range(1, np.max(labs)):
        x, y = np.where(labs == l)
        if len(x) < 100:
            continue
        points = np.array([x[::5], y[::5]]).transpose()
        dist2center = np.mean(euclidean_distances(center, points))
        if dist2center < dist:
            good_lab = l
            dist = dist2center
            
            
    return original, labs == good_lab
    
    
    