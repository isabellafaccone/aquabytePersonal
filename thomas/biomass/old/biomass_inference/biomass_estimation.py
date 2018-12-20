import json


class BiomassRegressionModel(object):

    def __init__(self):
        self.leading_coefficient = 56.703
        self.exponent = 3.2215

    def estimate_biomass(self, fish_detection_path, biomass_estimation_path):
        fish_detections = json.load(open(fish_detection_path, 'r'))
        biomass_estimates = []
        for fish_detection in fish_detections:
            total_length = fish_detection['total_length']
            biomass = self.leading_coefficient * ((0.01 * total_length) ** self.exponent)
            biomass_estimate = fish_detection.copy()
            biomass_estimate['biomass'] = biomass
            biomass_estimates.append(biomass_estimate)

        with open(biomass_estimation_path, 'w') as outfile:
            json.dump(biomass_estimates, outfile)