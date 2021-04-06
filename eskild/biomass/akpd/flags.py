#!/usr/bin/env python3

class Flags(object):
    def __init__(self,l_config, checkpoint_path):
        self.input_size = tuple(l_config["input_size"])
        self.stages = l_config["cpm_stages"]
        self.batch_size = l_config["batch_size"]
        self.joints = l_config["num_of_joints"]
        self.model_path = checkpoint_path
        self.cmap_radius = l_config["center_radius"]
        self.keypoints_order = l_config["keypoints_order"]
        self.normalize = l_config["normalize"]
        self.heatmap_size = 512
        self.joint_gaussian_variance = l_config["joint_gaussian_variance"]
        self.crop = l_config["crop"]
        self.augmentation = None