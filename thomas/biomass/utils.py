import numpy as np


FOCAL_LENGTH = 0.0084366
BASELINE = 0.128096
PIXEL_SIZE_M = 3.45 * 1e-6
FOCAL_LENGTH_PIXEL = FOCAL_LENGTH / PIXEL_SIZE_M
IMAGE_SENSOR_WIDTH = 0.01412
IMAGE_SENSOR_HEIGHT = 0.01035
PIXEL_COUNT_WIDTH = 4096
PIXEL_COUNT_HEIGHT = 3000

def convert_to_world_point(x, y, d):
    """ from pixel coordinates to world coordinates """
    
    image_center_x = PIXEL_COUNT_WIDTH / 2.0  
    image_center_y = PIXEL_COUNT_HEIGHT / 2.0
    px_x = x - image_center_x
    px_z = image_center_y - y

    sensor_x = px_x * (IMAGE_SENSOR_WIDTH / 4096)
    sensor_z = px_z * (IMAGE_SENSOR_HEIGHT / 3000)

    # d = depth_map[y, x]
    world_y = d
    world_x = (world_y * sensor_x) / FOCAL_LENGTH
    world_z = (world_y * sensor_z) / FOCAL_LENGTH
    return np.array([world_x, world_y, world_z])


def depth_from_disp(disp):
    """ calculate the depth of the point based on the disparity value """
    depth = FOCAL_LENGTH_PIXEL*BASELINE / np.array(disp)
    return depth


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5


def load_keypoints(annotation, mapping):
    """load keypoints coordinates to numpy array"""
    # the order is given by mapping
    order = mapping.keys()
    keypoints = []
    for kp_name in order:
        geometry = annotation["Label"][kp_name][0]["geometry"]
        keypoints.append([geometry["x"], geometry["y"]])
    keypoints = np.array(keypoints)
    return keypoints
