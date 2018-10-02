from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
#from utils import crop_and_mask
from obb import OBB
import cv2
import copy

def convert_to_world_point(x, y, depth_map, 
                           image_sensor_width=32.0*1e-3,
                           image_sensor_height = 18.0*1e-3,
                           focal_length = 10.0*1e-3):
    """
    Returns the real world coordinates of pixels (x, y) 
    given the depth map
    
    Input:
        - x, y: int, pixel coordinates
        - depth_map: np.array of size (W, H)
    
    Output:
        - world_x, world_y, world_z: tuple of int
    """
    image_center_x = 1024 / 2.0 #depth_map.shape[1] / 2.0
    image_center_y = 512 / 2.0 # depth_map.shape[0] / 2.0
    px_x = x - image_center_x
    px_z = image_center_y - y

    sensor_x = px_x * (image_sensor_width / 1024)
    sensor_z = px_z * (image_sensor_height / 512)
    
    d = depth_map[y, x]
    world_y = d
    world_x = (world_y * sensor_x) / focal_length
    world_z = (world_y * sensor_z) / focal_length
    
    return (world_x, world_y, world_z)

def compute_length(mask, depth_map):
    """
    Computes the length from mask & depth map
    
    Input:
         - mask: np.array of size (W, H)
         - depth_map: np.array of size (W, H)
         
    Output:
         - length: float
    """
    y, x = np.nonzero(depth_map * mask)
    wx, wy, wz = convert_to_world_point(x, y, depth_map)
    cloud = []
    for (i0, j0, k0) in zip(wx, wy, wz):
        cloud.append([i0, j0, k0])
    obb, eigen_vectors = OBB.build_from_points([(p[0], p[1], p[2])
                                                for p in cloud])
    true_obb_points = np.array(obb.points)
    true_length = np.linalg.norm(true_obb_points[0] - true_obb_points[1])
    
    return true_length

def compute_segmentation_error(modified_mask, mask):
    """
    Computes the error between a modified mask and a 
    ground truth mask
    
    Input: 
        - modified_mask: np.array of size (W, H)
        - mask: np.array of size (W, H)
    
    Output:
        - error: float
    """
    total = modified_mask + mask
    intersection = np.count_nonzero(total[total == 2])
    union = np.count_nonzero(total[total > 0])
    iou = intersection * 100 / union
    error = 100 - iou
    
    return error

def get_bb_from_mask(mask, mask_value=1):
    """
    Computes the bounding box coordinates from the mask
    
    Input:
        - mask : np.array of size (L, H, 3)
        - mask_value: float, 
    
    Output:
        - (x1, y1, x2, y2) : coordinates of the corner of
        the bounding box
    """
    x_end, x_start = (np.where(mask == mask_value)[0].max(),
                      np.where(mask == mask_value)[0].min())
    y_end, y_start = (np.where(mask == mask_value)[1].max(), 
                      np.where(mask == mask_value)[1].min())
    
    return (y_start, x_start, y_end, x_end)  

def computes_noised_sliced_dmap(mdepth, mask, x1, x2, stdev, nb_of_regions):
    """
    Returns a noised dmap per regions of the fish
    
    Inputs:
         - mdepth: np.array of size (H, W)
         - x1, x2: float, bb length from mask
         - stdev: float, standard deviation of gaussian noise
         - nb_of_regions: int, number of regions from the fish 
         mask
    
    Output:
        - noised_dmap: noised dmap per region, np.array (W, H)
    """
    new_depth = copy.deepcopy(mdepth)
    noised_dmap_list = []
    quotient = (x2 - x1) / nb_of_regions
    #print(x2, x1, quotient)
    for slice_ix in range(nb_of_regions):
        x_start = x1 + slice_ix * quotient
        if slice_ix != nb_of_regions - 1:
            x_end = x1 + (slice_ix + 1) *  quotient
        else:
            x_end = x2
        sliced_mdepth = new_depth[:, x_start:x_end]
        sliced_mask = mask[:, x_start:x_end]
        mean_slice_depth = sliced_mdepth.sum() / np.count_nonzero(sliced_mdepth)
        noise = np.zeros(sliced_mdepth.shape, np.uint8)
        cv2.randn(noise, np.array(0), np.ones(1) * stdev)
        #noise = np.random.normal(0, stdev, 1) * np.ones(sliced_mdepth.shape)
        sliced_mdepth += noise * sliced_mask
        noised_dmap_list.append(sliced_mdepth)
        
    noised_dmap = np.concatenate(noised_dmap_list, axis=1)
    r_noised_dmap = np.zeros(mask.shape)
    r_noised_dmap[:, x1:x2] = noised_dmap
    return r_noised_dmap
