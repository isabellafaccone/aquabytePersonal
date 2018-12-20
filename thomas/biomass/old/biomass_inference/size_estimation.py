import pickle
import json

import cv2
import numpy as np

from biomass_utils.quantities import Distance


class RegressionBasedAlgorithm(object):

    def __init__(self):
        pass

    def get_regression_based_endpoints(self, mask):
        mask_values = np.where(mask > 0)
        x_values = mask_values[1]
        y_values = mask_values[0]
        adj_y_values = mask.shape[0] - y_values

        A = np.vstack([x_values, np.ones(len(x_values))]).T
        res = np.linalg.lstsq(A, adj_y_values)
        m, b = res[0]

        # get length endpoints
        x_lower = 0
        while x_lower < mask.shape[1]:
            adj_y_lower = int(round(m * x_lower + b))
            if x_lower in x_values and adj_y_lower in adj_y_values:
                break
            x_lower += 1

        x_upper = mask.shape[1]
        while x_upper > 0:
            adj_y_upper = int(round(m * x_upper + b))
            if x_upper in x_values and adj_y_upper in adj_y_values:
                break
            x_upper -= 1

        y_lower = mask.shape[0] - adj_y_lower
        y_upper = mask.shape[0] - adj_y_upper
        length_endpoint_1 = (x_lower, y_lower)
        length_endpoint_2 = (x_upper, y_upper)

        # get width endpoints
        m = -1 / float(m)
        b = adj_y_values.mean() - m * x_values.mean()

        adj_y_lower = 0
        while adj_y_lower < mask.shape[0]:
            x_lower = int(round((adj_y_lower - b) / float(m)))
            if x_lower in x_values and adj_y_lower in adj_y_values:
                break
            adj_y_lower += 1

        adj_y_upper = mask.shape[0]
        while adj_y_upper > 0:
            x_upper = int(round((adj_y_upper - b) / float(m)))
            if x_upper in x_values and adj_y_upper in adj_y_values:
                break
            adj_y_upper -= 1

        y_lower = mask.shape[0] - adj_y_lower
        y_upper = mask.shape[0] - adj_y_upper
        width_endpoint_1 = (x_lower, y_lower)
        width_endpoint_2 = (x_upper, y_upper)

        # get centroid coordinates

        x_centroid = mask_values[1].mean()
        y_centroid = mask_values[0].mean()
        centroid = (x_centroid, y_centroid)

        return {
            'length_endpoint_1': length_endpoint_1,
            'length_endpoint_2': length_endpoint_2,
            'width_endpoint_1': width_endpoint_1,
            'width_endpoint_2': width_endpoint_2,
            'centroid': centroid
        }

    def convert_to_world_point(self, point_in_pixels, frame_pixel_dimensions, depth, camera_parameters):
        pixel_x, pixel_y = point_in_pixels
        adj_pixel_y = frame_pixel_dimensions[1] - pixel_y
        origin = (int(frame_pixel_dimensions[0] / 2), int(frame_pixel_dimensions[1] / 2))
        world_x = depth.represented_as('cm') * (pixel_x - origin[0]) * camera_parameters[
            'effective_pixel_width'].represented_as('cm') / (camera_parameters['focal_length'].represented_as('cm'))
        world_y = depth.represented_as('cm') * (adj_pixel_y - origin[1]) * camera_parameters[
            'effective_pixel_height'].represented_as('cm') / (camera_parameters['focal_length'].represented_as('cm'))
        return (world_x, world_y)

    def generate_dimensional_metrics(self, mask, depth, camera_parameters):
        data = self.get_regression_based_endpoints(mask)

        frame_pixel_dimensions = (mask.shape[1], mask.shape[0])
        length_endpoint_1 = self.convert_to_world_point(data['length_endpoint_1'], frame_pixel_dimensions, depth,
                                                        camera_parameters)
        length_endpoint_2 = self.convert_to_world_point(data['length_endpoint_2'], frame_pixel_dimensions, depth,
                                                        camera_parameters)
        width_endpoint_1 = self.convert_to_world_point(data['width_endpoint_1'], frame_pixel_dimensions, depth,
                                                       camera_parameters)
        width_endpoint_2 = self.convert_to_world_point(data['width_endpoint_2'], frame_pixel_dimensions, depth,
                                                       camera_parameters)
        centroid = self.convert_to_world_point(data['centroid'], frame_pixel_dimensions, depth, camera_parameters)

        total_length = Distance(((length_endpoint_1[0] - length_endpoint_2[0]) ** 2 + (
                    length_endpoint_1[1] - length_endpoint_2[1]) ** 2) ** 0.5, 'cm')
        total_width = Distance(((width_endpoint_1[0] - width_endpoint_2[0]) ** 2 + (
                    width_endpoint_1[1] - width_endpoint_2[1]) ** 2) ** 0.5, 'cm')

        return {
            'total_length': total_length,
            'total_width': total_width,
            'centroid': centroid
        }

    def generate_fish_detections(self, frame_segmentation_path, disparity_matrix_path, fish_detection_path,
                                 camera_config):

        camera_parameters = {
            'baseline': Distance(camera_config['baseline'], 'cm'),
            'focal_length': Distance(camera_config['focal_length_air'], 'mm'),
            'effective_pixel_width': Distance(camera_config['effective_pixel_width'], 'um'),
            'effective_pixel_height': Distance(camera_config['effective_pixel_width'], 'um')
        }

        fish_segmentations = pickle.load(open(frame_segmentation_path, 'rb'))
        disparity_matrix = pickle.load(open(disparity_matrix_path, 'rb'))
        fish_detections = []

        for fish_segmentation in fish_segmentations:
            fish_segmentation_mask = fish_segmentation['segmentation']
            masked_disparity_matrix = cv2.bitwise_and(disparity_matrix, disparity_matrix, mask=fish_segmentation_mask)
            masked_disparity = masked_disparity_matrix[masked_disparity_matrix > 0]
            if len(masked_disparity) > 0:
                # get depth of fish
                avg_masked_pixel_disparity = np.mean(masked_disparity)
                avg_masked_disparity = Distance(
                    avg_masked_pixel_disparity * camera_parameters['effective_pixel_width'].represented_as('cm'), 'cm')
                depth = Distance((camera_parameters['baseline'].represented_as('cm') * camera_parameters[
                    'focal_length'].represented_as('cm')) / avg_masked_disparity.represented_as('cm'), 'cm')

                # get regression based length endpoints and width endpoints
                dimensional_data = self.generate_dimensional_metrics(masked_disparity_matrix, depth, camera_parameters)
                total_length = dimensional_data['total_length'].represented_as('cm')
                total_width = dimensional_data['total_width'].represented_as('cm')
                centroid = dimensional_data['centroid']
                box_area = total_length * total_width

                fish_detection = {
                    'centroid_x_position': centroid[0],
                    'centroid_y_position': centroid[1],
                    'total_length': total_length,
                    'total_width': total_width,
                    'box_area': box_area,
                    'depth': depth.represented_as('cm')
                }

                fish_detections.append(fish_detection)

        with open(fish_detection_path, 'w') as outfile:
            json.dump(fish_detections, outfile)


def main():
    frame_segmentation_path = '/home/paperspace/aquabyte/aquabyte_ml/test/temp/test_run_v1/a805f6e55638bbe362e3745ea61491e52a13a36f/frame_segmentation.pkl'
    disparity_matrix_path = '/home/paperspace/aquabyte/aquabyte_ml/test/temp/test_run_v1/51f003f816b04c97d43cd8f3c6943395754f3b02/disparity_matrix.pkl'
    fish_detection_path = '/tmp/fish_detections.pkl'
    config_path = '/home/paperspace/aquabyte/aquabyte_ml/config/config.json'

    camera_config = json.load(open(config_path))['test']['camera_config']

    rga = RegressionBasedAlgorithm()
    rga.generate_fish_detections(frame_segmentation_path, disparity_matrix_path, fish_detection_path, camera_config)


if __name__ == '__main__':
    main()





