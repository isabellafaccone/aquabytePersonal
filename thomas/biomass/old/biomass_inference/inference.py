import json

from biomass_estimation import BiomassRegressionModel
from size_estimation import RegressionBasedAlgorithm


class BiomassInference:
    """wrapper of size estimation and biomass estimation"""

    def __init__(self, config_path):
        """nothing going here for now"""
        self.rga = RegressionBasedAlgorithm()
        self.brm = BiomassRegressionModel()
        self.camera_config = json.load(open(config_path))['test']['camera_config']

    def run_batch(self, frame_segmentation_path, disparity_matrix_path, fish_detection_path, camera_config,
                  biomass_estimation_path):
        """run batch of size estimation followed by biomass estimation"""
        # fish size estimation
        self.rga.generate_fish_detections(frame_segmentation_path, disparity_matrix_path, fish_detection_path,
                                          camera_config)
        # biomass estimation
        self.brm.estimate_biomass(fish_detection_path, biomass_estimation_path)
