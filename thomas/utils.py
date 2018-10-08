import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


FOCAL_LENGTH = 0.0107
BASELINE = 0.135
PIXEL_SIZE_M = 3.45 * 1e-6
FOCAL_LENGTH_PIXEL = FOCAL_LENGTH / PIXEL_SIZE_M
IMAGE_SENSOR_WIDTH = 0.01412
IMAGE_SENSOR_HEIGHT = 0.01412
RANDOM_STATE = 170
DENSITY = 1e6  # g per cubic meter

LENGTH_FACTOR = 44.0  # where is this coming from? DATA
COST_THRESHOLD = 100.0  # another magic number


def convert_to_world_point(x, y, d):
    """ from pixel coordinates to world coordinates """
    # TODO (@Thomas) this is hard coded and this bad....
    image_center_x = 3000 / 2.0  # depth_map.shape[1] / 2.0
    image_center_y = 4096 / 2.0  # depth_map.shape[0] / 2.0
    px_x = x - image_center_x
    px_z = image_center_y - y

    sensor_x = px_x * (IMAGE_SENSOR_WIDTH / 3000)
    sensor_z = px_z * (IMAGE_SENSOR_HEIGHT / 4096)

    # d = depth_map[y, x]
    world_y = d
    world_x = (world_y * sensor_x) / FOCAL_LENGTH
    world_z = (world_y * sensor_z) / FOCAL_LENGTH
    return (world_x, world_y, world_z)


def left_right_matching(left_annotations, right_annotations):
    """match the bboxes. Return a list of matched bboxes"""
    # let's use x1, x2 to match bboxes
    left_centroids = []
    for ann in left_annotations:
        bbox = ann['bbox']
        # centroid = [(bbox[3] - bbox[1])/2.0, (bbox[2] - bbox[0])/2.0]
        centroid = [bbox[2], bbox[0]]
        left_centroids.append(centroid)

    right_centroids = []
    for ann in right_annotations:
        bbox = ann['bbox']
        # centroid = [(bbox[3] - bbox[1])/2.0, (bbox[2] - bbox[0])/2.0]
        centroid = [bbox[2], bbox[0]]
        right_centroids.append(centroid)

    print("Number of left centroids: {}".format(len(left_centroids)))
    print("Number of right centroids: {}".format(len(right_centroids)))

    # euclidean distance in (x1, x2) space
    cost_matrix = euclidean_distances(left_centroids, right_centroids)

    # hungarian algorithm to minimize weights in bipartite graph
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_annotations = []
    for (r, c) in zip(row_ind, col_ind):
        if cost_matrix[r, c] < COST_THRESHOLD:
            matched_annotations.append([left_annotations[r], right_annotations[c]])

    return matched_annotations


def weight_estimator(left_annotation, right_annotation):
    """
    estimate weights from left and right masks
    inputs:
    - left annotation (COCO format)
    - right annotation (COCO format)
    outputs:
    - weight
    """
    # calculing the kmeans centroid for the left mask
    seg = left_annotation['segmentation'][0]
    left_poly = np.array(seg).reshape((int(len(seg) / 2), 2))
    lp = [(r[0], r[1]) for r in left_poly]
    left_mask = Image.new('L', (4096, 3000), 0)
    ImageDraw.Draw(left_mask).polygon(lp, outline=1, fill=1)
    left_mask = np.array(left_mask)
    left_X = np.stack([np.nonzero(left_mask)[0], np.nonzero(left_mask)[1]], axis=1)
    left_X = left_X[::100, :]  # downsample

    left_y_pred = KMeans(n_clusters=6, random_state=RANDOM_STATE).fit_predict(left_X)
    centroids = []
    for label in np.unique(left_y_pred):
        x_mean = np.mean(left_X[left_y_pred==label, 0])
        y_mean = np.mean(left_X[left_y_pred==label, 1])
        centroids.append((x_mean, y_mean))
    left_centroids = np.array(centroids)
    left_centroids = left_centroids[left_centroids[:,1].argsort()]
    print(left_centroids)
    
    # calculing the kmeans centroid for the right mask
    seg = right_annotation['segmentation'][0]
    right_poly = np.array(seg).reshape((int(len(seg) / 2), 2))
    #     right_X = right_poly
    rp = [(r[0], r[1]) for r in right_poly]
    right_mask = Image.new('L', (4096, 3000), 0)
    ImageDraw.Draw(right_mask).polygon(rp, outline=1, fill=1)
    right_mask = np.array(right_mask)
    right_X = np.stack([np.nonzero(right_mask)[0], np.nonzero(right_mask)[1]], axis=1)
    right_X = right_X[::100, :]

    right_y_pred = KMeans(n_clusters=6, random_state=RANDOM_STATE).fit_predict(right_X)
    centroids = []
    for label in np.unique(right_y_pred):
        x_mean = np.mean(right_X[right_y_pred == label, 0])
        y_mean = np.mean(right_X[right_y_pred == label, 1])
        centroids.append((x_mean, y_mean))
    right_centroids = np.array(centroids)
    right_centroids = right_centroids[right_centroids[:, 1].argsort()]
    print(right_centroids)
    
    # calculate the disparties between the left and right most centroids
    disparities = left_centroids[[0, -1], 1] - right_centroids[[0, -1], 1]
    print(disparities)
    
    # and the depth
    depth = FOCAL_LENGTH_PIXEL * BASELINE / np.array(disparities)

    # now calculate the world coordinates and the length
    world0 = convert_to_world_point(left_centroids[0][0], left_centroids[0][1], depth[0])
    world1 = convert_to_world_point(left_centroids[-1][0], left_centroids[-1][1], depth[1])
    predicted_length = np.linalg.norm(np.array(world0) - np.array(world1))

    # hack to remove the long length
    if predicted_length > 1.5:
        return
    else:
        # let's hack it into a volume
        biomass = predicted_length ** 3 * DENSITY / LENGTH_FACTOR  # the length_factor factor is purely empirical
        return biomass
