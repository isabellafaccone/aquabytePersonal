import json
import math
import numpy as np
import statsmodels.api as sm
from PIL import Image, ImageDraw


# get distance between two points in R3
def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# get distance between two pixels
def distance_between_points_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  


# get a point cloud of the full image (i.e a matrix where each entry gives the world coordinates corresponding to that pixel)
def get_point_cloud(blender_depth_map, focal_length, sensor_width, sensor_height):
    
    px_count_width = blender_depth_map.shape[1]
    px_count_height = blender_depth_map.shape[0]
    
    image_center_x = int(px_count_width / 2)
    image_center_y = int(px_count_height / 2)

    a = np.tile(np.array(range(blender_depth_map.shape[1])), [blender_depth_map.shape[0], 1])
    b = px_count_height - 1 - np.tile(np.array(range(blender_depth_map.shape[0])), [blender_depth_map.shape[1], 1]).T
    pixel_array = np.dstack([b, a]) - np.array([image_center_y, image_center_x])

    sensor_array = pixel_array * np.array([(sensor_height / px_count_height), (sensor_width / px_count_width)])


    world_y = blender_depth_map / np.sqrt(1 + (sensor_array[:,:,0]**2 + sensor_array[:,:,1]**2) / (focal_length**2))
    world_x = (sensor_array[:,:,1] * world_y) / focal_length
    world_z = (sensor_array[:,:,0] * world_y) / focal_length
    point_cloud = np.dstack([world_x, world_y, world_z])
    
    return point_cloud


# get the annotation file, mask, blender depth map, and point cloud for a given pair of stereo images
def get_data(data_dir_base, idx, include_image=True):
    # get annotation data
    annotation_file_name = 'annot_{}.json'.format(idx)
    annotation_file_path = '{}/{}/{}'.format(data_dir_base, 'annotations', annotation_file_name)
    annotation_data = json.load(open(annotation_file_path, 'rb'))
    print(annotation_data)

    # get segmentation data
    segmentation_file_name = 'left_{}.npy'.format(idx)
    segmentation_file_path = '{}/{}/{}'.format(data_dir_base, 'mask', segmentation_file_name)
    mask = np.load(segmentation_file_path)

    # get depth map data
    blender_depth_map_file_name = 'depth_map_{}.npy'.format(idx)
    blender_depth_map_file_path = '{}/{}/{}'.format(data_dir_base, 'depth_map', blender_depth_map_file_name)
    blender_depth_map = 10 * np.load(blender_depth_map_file_path).T # the multiplication by 10 is to convert from dm to cm

    # convert blender depth map to orthogonal depth map (i.e. a map of the distance between each point and the plane containing the two camera sensors)
    point_cloud = get_point_cloud(blender_depth_map, annotation_data['focal_length'], annotation_data['sensor_height'], annotation_data['sensor_width'])
    
    # get image
    data = { 'annotation_data': annotation_data, 'mask': mask, 'blender_depth_map': blender_depth_map, 'point_cloud': point_cloud }
    if include_image:
        image_file_name = 'left_{}.png'.format(idx)
        image_file_path = '{}/{}/{}'.format(data_dir_base, 'stereo_images', image_file_name)
        image = Image.open(image_file_path)
        data['image'] = image
        
    return data


# determine the cutoff for depth values on the fish's surface (this is to account for the fact that some points near the boundary of the mask
# are not on the fish itself and end up with an unacceptably large depth value. The cutoff is such that if a pixel near the boundary of the mask
# has a depth value that exceeds this cutoff, it is not considered to not be a part of the fish - it is a part of the back wall
def get_depth_cutoff(depth_map, mask):
	hist_counts, hist_bucket_endpoints = np.histogram(depth_map[np.where(mask > 0)], 20)
	cutoff_idx = np.argmin(hist_counts)
	cutoff = hist_bucket_endpoints[cutoff_idx]
	return cutoff


# get the length enpoints, width endpoints, and visible centroid in pixel coordinates (we obtain this using the linear regression technique 
# described in Alok's research diary -- see the drive or ask him about it)
def get_points_of_interest(mask, depth_map, cutoff):
    mask_values = np.where(mask > 0)
    x_values = mask_values[1]
    y_values = mask_values[0]
    adj_y_values = mask.shape[0] - y_values
    mask_points = list(zip(x_values, adj_y_values))

    A = np.vstack([x_values, np.ones(len(x_values))]).T
    res = np.linalg.lstsq(A,adj_y_values)
    m, b = res[0]

    # get length endpoints
    x_lower = x_values.min()
    while x_lower < mask.shape[1]:
        adj_y_lower = int(round(m*x_lower + b))
        y_lower = mask.shape[0] - 1 - adj_y_lower
        if ((x_lower, adj_y_lower) in mask_points and (depth_map[y_lower, x_lower] < cutoff)): 
            break
        x_lower += 1

    x_upper = x_values.max()
    while x_upper > 0:
        adj_y_upper = int(round(m*x_upper + b))
        y_upper = mask.shape[0] - 1 - adj_y_upper
        if ((x_upper, adj_y_upper) in mask_points and (depth_map[y_upper, x_upper] < cutoff)):
            break
        x_upper -= 1

    
    length_endpoint_1 = (y_lower, x_lower)
    length_endpoint_2 = (y_upper, x_upper)

    # get width endpoints
    m = -1 / float(m)
    b = adj_y_values.mean() - m*x_values.mean()

    adj_y_lower = adj_y_values.min()
    while adj_y_lower < mask.shape[0]:
        x_lower = int(round((adj_y_lower - b)/float(m)))
        y_lower = mask.shape[0] - 1 - adj_y_lower
        if ((x_lower, adj_y_lower) in mask_points and (depth_map[y_lower, x_lower] < cutoff)):
            break
        adj_y_lower += 1


    adj_y_upper = adj_y_values.max()
    while adj_y_upper > 0:
        x_upper = int(round((adj_y_upper - b)/float(m)))
        y_upper = mask.shape[0] - 1 - adj_y_upper
        if ((x_upper, adj_y_upper) in mask_points and (depth_map[y_upper, x_upper] < cutoff)):
            break
        adj_y_upper -= 1

    width_endpoint_1 = (y_lower, x_lower)
    width_endpoint_2 = (y_upper, x_upper)

    # get centroid coordinates
    x_visible_centroid = mask_values[1].mean()
    y_visible_centroid = mask_values[0].mean()
    visible_centroid = (int(round(y_visible_centroid)), int(round(x_visible_centroid)))
    return {
        'length_endpoint_1': length_endpoint_1,
        'length_endpoint_2': length_endpoint_2,
        'width_endpoint_1': width_endpoint_1,
        'width_endpoint_2': width_endpoint_2,
        'visible_centroid': visible_centroid
    }








