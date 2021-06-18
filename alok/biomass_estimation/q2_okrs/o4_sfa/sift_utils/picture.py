import json
import os
import copy
import cv2
import numpy as np
from PIL import Image
from research.utils.data_access_utils import S3AccessUtils

s3 = S3AccessUtils('/root/data', json.load(open(os.environ['AWS_CREDENTIALS'])))


class Picture(object):
    def __init__(self, image_url=None, image_arr=None):
        self.image_url = image_url
        if image_url:
            image_f, _, _ = s3.download_from_url(image_url)
            image = Image.open(image_f)
            self.image_arr = np.array(image)
        else:
            self.image_arr = image_arr
        self.image = Image.fromarray(self.image_arr)

    # apply CLAHE (contrast limited adaptive histogram equalization) as well as image sharpening
    def enhance(self, clip_limit=5, tile_grid_size=(8, 8), in_place=True, sharpen=True, sharp_grid_size=(21, 21),
                sharpen_weight=2.0):
        # convert image to LAB color model
        image_lab = cv2.cvtColor(self.image_arr, cv2.COLOR_BGR2LAB)

        # split the image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(image_lab)

        # apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L channel with the original A and B channel
        merged_channels = cv2.merge((cl, a_channel, b_channel))

        # convert image from LAB color model back to RGB color model
        enhanced_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)

        if sharpen:
            blurred = cv2.GaussianBlur(enhanced_image, sharp_grid_size, 0)
            enhanced_image = cv2.addWeighted(enhanced_image, sharpen_weight, blurred, 1-sharpen_weight, 0)

        if in_place:
            self.image_arr = enhanced_image
            self.image = Image.fromarray(self.image_arr)
        else:
            return Picture(image_arr=enhanced_image)

    # return crop of image (here, x-axis is horizontal axis, y-axis is vertical axis. The order
    # of coordinate references for numpy array versions of images is y, x)
    def generate_crop(self, x_min, y_min, width, height, return_copy=False):
        x_max = x_min + width
        y_max = y_min + height
        assert x_min >= 0 and x_max < self.image_arr.shape[1] and y_min >= 0 and y_max < self.image_arr.shape[0], \
               'bounding box coordinates are out-of-bounds'
        crop = self.image_arr[y_min:y_max, x_min:x_max]
        if return_copy:
            return copy.copy(crop)
        return crop

    def get_image_arr(self):
        return self.image_arr

    def get_image(self):
        return self.image
