import numpy as np
import cv2

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

def convert_to_world_point(x, y, depth_map,
                           image_sensor_width=3.2,
                           image_sensor_height=1.8,
                           focal_length=1.0):
    """
    This function need a fix @bryton: incoherent values
    For isntance test: x1=10, y1=0, x2=40, y2=0
    Returns the real world coordinates of pixels (x, y) 
    given the depth map

    Input:
        - x, y: int, pixel coordinates
        - depth_map: np.array of size (W, H)

    Output:
        - world_x, world_y, world_z: tuple of int
    """
    px_count_width = depth_map.shape[1]
    px_count_height = depth_map.shape[0]
    image_center_x = depth_map.shape[1] / 2.0
    image_center_y = depth_map.shape[0] / 2.0
    px_x = x - image_center_x
    px_z = image_center_y - y
    sensor_x = px_x * (image_sensor_width / px_count_width)
    sensor_z = px_z * (image_sensor_height / px_count_height)
    d = depth_map[y, x]
    world_y = d
    world_x = (world_y * sensor_x) / focal_length
    world_z = (world_y * sensor_z) / focal_length
    
    return (world_x, world_y, world_z)

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