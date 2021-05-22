from collections import namedtuple
import math
from typing import Dict, List, Tuple
import numpy as np


BODY_PARTS = [
    'ADIPOSE_FIN',
    'ANAL_FIN',
    'DORSAL_FIN',
    'EYE',
    'PECTORAL_FIN',
    'PELVIC_FIN',
    'TAIL_NOTCH',
    'UPPER_LIP'
]


CameraMetadata = namedtuple('CameraMetadata',
                            ['focal_length', 'focal_length_pixel', 'baseline_m',
                             'pixel_count_width', 'pixel_count_height', 'image_sensor_width',
                             'image_sensor_height'])


def deg_to_rad(theta_deg):
    theta_rad = theta_deg * np.pi / 180.0
    return theta_rad


def rad_to_deg(theta_rad):
    theta_deg = theta_rad * 180.0 / np.pi
    return theta_deg


def get_2D_coords_from_ann(annotation: dict) -> tuple:
    """Gets numpy array of left and right keypoints given input keypoint annotation."""

    left_keypoints, right_keypoints = {}, {}
    for item in annotation['leftCrop']:
        body_part = item['keypointType']
        left_keypoints[body_part] = (item['xFrame'], item['yFrame'])

    for item in annotation['rightCrop']:
        body_part = item['keypointType']
        right_keypoints[body_part] = (item['xFrame'], item['yFrame'])

    left_keypoint_arr, right_keypoint_arr = [], []
    for body_part in BODY_PARTS:
        left_keypoint_arr.append(left_keypoints[body_part])
        right_keypoint_arr.append(right_keypoints[body_part])

    X_left = np.array(left_keypoint_arr)
    X_right = np.array(right_keypoint_arr)
    return X_left, X_right


def get_3D_coords_from_2D(X_left: np.ndarray, X_right: np.ndarray,
                               camera_metadata: CameraMetadata) -> np.ndarray:
    """Converts input left and right 2D coords into 3D coords."""

    y_world = camera_metadata.focal_length_pixel * camera_metadata.baseline_m / \
              (X_left[:, 0] - X_right[:, 0])

    x_world = (X_left[:, 0] - 0.5*camera_metadata.pixel_count_width) * y_world / camera_metadata.focal_length_pixel
    z_world = -(X_left[:, 1] - 0.5*camera_metadata.pixel_count_height) * y_world / camera_metadata.focal_length_pixel
    X_world = np.vstack([x_world, y_world, z_world]).T
    return X_world


def get_2D_coords_from_3D(coords: np.ndarray, camera_metadata: CameraMetadata) -> np.ndarray:
    """Converts input 3D coords into left and right 2D coords."""

    X_left, X_right = [], []
    for i in range(coords.shape[0]):
        disparity = camera_metadata.focal_length_pixel * camera_metadata.baseline_m / coords[i][1]
        x_left = coords[i][0] * disparity / camera_metadata.baseline_m + 0.5*camera_metadata.pixel_count_width
        y_left = -coords[i][2] * disparity / camera_metadata.baseline_m + 0.5*camera_metadata.pixel_count_height
        x_right = x_left - disparity
        y_right = y_left
        X_left.append([x_left, y_left])
        X_right.append([x_right, y_right])
    X_left = np.array(X_left)
    X_right = np.array(X_right)
    return X_left, X_right


def R_from_euler_angles(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Generates the matrix for rotation of angle theta about an axis aligned with unit vector n.

    Inputs:
    - alpha: yaw about z-axis; beta: pitch about y' axis; gamma: roll about x'' axis
    Outputs:
    - 3x3 rotation matrix
    
    """

    R_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def translate_coordinates(coords: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply translation such that medoid becomes the new origin."""

    translated_coords = coords + v
    return translated_coords

    
def rotate_and_reposition(coords: np.ndarray, alpha: float, beta: float, 
                          gamma: float, new_position: np.ndarray) -> np.ndarray:
    """Center coordinates, rotate by euler angles, and reposition to new location. """
    
    centered_coords = coords - np.median(coords, axis=0)
    R = R_from_euler_angles(alpha, beta, gamma)
    rotated_coords = np.dot(R, centered_coords.T).T
    repositioned_coords = rotated_coords + new_position
    return repositioned_coords


def jitter_2D_coords(X_left, X_right, jitter_std):
    X_left_jittered = np.empty_like(X_left)
    X_left_jittered[:] = X_left
    X_left_jittered[:, 0] += np.random.normal(0, jitter_std, X_left.shape[0])

    X_right_jittered = np.empty_like(X_right)
    X_right_jittered[:] = X_right
    X_right_jittered[:, 0] += np.random.normal(0, jitter_std, X_right.shape[0])
    
    return X_left_jittered, X_right_jittered


def generate_rotation_matrix(n, theta):
    """generate the matrix for rotation of angle theta about an axis aligned with unit vector n."""
    R = np.array([[
        np.cos(theta) + n[0] ** 2 * (1 - np.cos(theta)),
        n[0] * n[1] * (1 - np.cos(theta)) - n[2] * np.sin(theta),
        n[0] * n[2] * (1 - np.cos(theta)) + n[1] * np.sin(theta)
    ], [
        n[1] * n[0] * (1 - np.cos(theta)) + n[2] * np.sin(theta),
        np.cos(theta) + n[1] ** 2 * (1 - np.cos(theta)),
        n[1] * n[2] * (1 - np.cos(theta)) - n[0] * np.sin(theta),
    ], [
        n[2] * n[0] * (1 - np.cos(theta)) - n[1] * np.sin(theta),
        n[2] * n[1] * (1 - np.cos(theta)) + n[0] * np.sin(theta),
        np.cos(theta) + n[2] ** 2 * (1 - np.cos(theta))
    ]])

    return R



def center_3D_coordinates(coords):
    """apply rotation to 3D coordinates about origin such that
    centroid is on positive y-axis (i.e. camera is looking straight at it).""" 

    v = np.median(coords, axis=0)
    v /= np.linalg.norm(v)
    y = np.array([0, 1, 0])
    n = np.cross(y, v)
    if np.linalg.norm(n) > 0:
        n /= np.linalg.norm(n)
        theta = -np.arccos(np.dot(y, v))
        R = generate_rotation_matrix(n, theta)
        rotated_coords = np.dot(R, coords.T).T
        return rotated_coords
    return coords



# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler_angles(R) :
    assert(is_rotation_matrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[1,0], R[0,0])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[2,1] , R[2,2])
    else :
        x = 0
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(-R[1,2], R[1,1])

    return [x, y, z]




