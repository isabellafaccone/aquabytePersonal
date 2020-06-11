from research.utils.data_access_utils import RDSAccessUtils, S3AccessUtils
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2

LICE_BBOX_COLOR = ['b', 'r'] # bbox edge color
LICE_CATEGORY = ['ADULT_FEMALE', 'MOVING']


class Visualizer(object):

    def __init__(self, s3_access_utils, rds_access_utils):

        self.s3_access_utils = s3_access_utils
        self.rds_access_utils = rds_access_utils
        self.image = None
        self.annotation = None
        self.fig = None
        self.ax = None
    
    
    def load_image(self, fish):
        self.annotation = fish['annotation']
        left_image_f, _, _ = self.s3_access_utils.download_from_url(fish['left_crop_url'])
        
        alpha, beta = 2, 15 # Contrast(1.0-3.0), Brightness(0-100)
        img = Image.open(left_image_f)
        img = np.asarray(img)
        self.image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        self.fig, self.ax = plt.subplots(1, figsize=(12, 20))


        
    def display_lice(self, lice, x, y, ax = None):
        lp = lice['position'] 
        w, h = lp["width"], lp["height"]
        class_index = LICE_CATEGORY.index(lice['category'])
        ec = LICE_BBOX_COLOR[class_index]
        rect = patches.Rectangle((x, y), w, h,linewidth=1,edgecolor=ec,facecolor='none')    
        if not ax:
            self.ax.add_patch(rect)
        else:
            ax.add_patch(rect)
    
    def display_crop(self, x, y , w, h, location):
        location_index = ["TOP", "MIDDLE", "BOTTOM"].index(location)
        ec = ["yellow", "green", "WHITE"][location_index]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, facecolor='none', edgecolor = ec)
        self.ax.add_patch(rect)

        
    def show_crops(self, crops):
        for crop in crops:
            fig, ax = plt.subplots(figsize=(10, 10))

            crop_left, crop_top = crop
            cropped_image = self.image[crop_top:(crop_top + 512), crop_left:(crop_left + 512)]
            for lice in crops[crop]:
                x = lice['position']['left'] - crop_left
                y = lice['position']['top'] - crop_top
                self.display_lice(lice, x, y, ax)
            ax.imshow(cropped_image)

    def show(self):
        self.ax.imshow(self.image)


