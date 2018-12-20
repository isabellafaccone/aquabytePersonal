import glob
import os

import numpy as np
from scipy.spatial import ConvexHull


focal_length = 0.0107
baseline = 0.135
pixel_size_m = 3.45 * 1e-6 
focal_length_pixel = focal_length / pixel_size_m
image_sensor_width = 0.01412
image_sensor_height = 0.01412


def depth_from_disp(disp):
    depth = focal_length_pixel*baseline / np.array(disp)
    return depth


def convert_to_world_point(x, y, d):
    image_center_x = 3000 / 2.0 #depth_map.shape[1] / 2.0
    image_center_y = 4096 / 2.0# depth_map.shape[0] / 2.0
    px_x = x - image_center_x
    px_z = image_center_y - y

    sensor_x = px_x * (image_sensor_height / 3000)
    sensor_z = px_z * (image_sensor_width / 4096)
    
    # d = depth_map[y, x]
    world_y = d
    world_x = (world_y * sensor_x) / focal_length
    world_z = (world_y * sensor_z) / focal_length
    return np.array((world_x, world_y, world_z))


def get_local_dic():
    all_images_path = glob.glob('/root/data/small_pen_data_collection/*_rectified/*.jpg')
    local_dic = {}
    for path in all_images_path:
        if 'rectified' not in path:
            local_dic[os.path.basename(path)] = path
    return local_dic


def get_pairs(example_coco, local_dic=None):
    # create pairs list
    pairs = {}
    for (imgid, imgdata) in example_coco.imgs.items():
        if 'local_path' in imgdata:
            img_path = imgdata['local_path']
        else:
            file_name = imgdata['coco_url'].split('%2F')[2].split('?alt')[0]
            img_path = local_dic[file_name]
        annotation_ids = example_coco.getAnnIds(imgIds=[imgid])
        if len(annotation_ids) == 0:
            continue
        if 'rectified' in img_path:
            ts = os.path.basename(img_path).split('.')[0].split('_')[-1]
            side = os.path.basename(img_path).split('.')[0].split('_')[0]
            if ts not in pairs:
                pairs[ts] = {}
            pairs[ts][side] = imgid
    return pairs


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval