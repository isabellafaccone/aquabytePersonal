import sys
import os
import cjson

sys.path.append('aquabyte_python')
sys.path.append(os.path.abspath("lib/fastmask/caffe-fm/python"))
sys.path.append(os.path.abspath("lib/fastmask/python_layers"))
sys.path.append(os.path.abspath("lib/fastmask"))
sys.path.append(os.path.abspath("lib"))
print sys.path

import caffe
from IPython import embed
from fastmask import config

import numpy as np
# import setproctitle
import cv2

from alchemy.utils.image import resize_blob, visualize_masks, load_image
from alchemy.utils.timer import Timer
from alchemy.utils.mask import encode, decode, crop, iou
from alchemy.utils.load_config import load_config

from fastmask.utils import gen_masks
from os import listdir
from os.path import isfile, join

from includes import session, Frame, FrameSegmentation
from frame_segmentation_algorithm import FrameSegmentationAlgorithm
import json
import datetime as dt
from utils.s3_utils import download_file_contents_from_s3_static


COLORS = [0xE6E2AF, 0xA7A37E, 0xDC3522, 0x046380, 
        0x468966, 0xB64926, 0x8E2800, 0xFFE11A,
        0xFF6138, 0x193441, 0xFF9800, 0x7D9100,
        0x1F8A70, 0x7D8A2E, 0x2E0927, 0xACCFCC,
        0x644D52, 0xA49A87, 0x04BFBF, 0xCDE855,
        0xF2836B, 0x88A825, 0xFF358B, 0x01B0F0,
        0xAEEE00, 0x334D5C, 0x45B29D, 0xEFC94C,
        0xE27A3F, 0xDF5A49]




class FastMaskAlgorithm(FrameSegmentationAlgorithm):

    def __init__(self, parameters):
        self.parameters = json.loads(parameters)
        self.model = str(self.parameters['model'])
        self.init_weights = str(self.parameters['init_weights'])
        self.threshold = self.parameters['threshold']
        self.net = None
        self.load_net()

    def load_net(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)

        self.net = caffe.Net(
            'lib/fastmask/models/' + self.model + '.test.prototxt',
            'lib/fastmask/params/' + self.init_weights,
            caffe.TEST)

        # load configgst
        if os.path.exists('lib/fastmask/configs/{}.json'.format(self.model)):
            load_config('lib/fastmask/configs/{}.json'.format(self.model))
        else:
            print "Specified config does not exists, use the default config..."

    def generate_frame_segmentation(self, image_file):
        
        image = cv2.imread(image_file).astype(np.float64)

        oh, ow = image.shape[:2]
        im_scale = config.TEST_SCALE * 1.0 / max(oh, ow)
        input_blob = image - config.RGB_MEAN
        input_blob = input_blob.transpose((2, 0, 1))
        ih, iw = int(oh * im_scale), int(ow * im_scale)
        ih, iw = ih - ih % 4, iw - iw % 4
        input_blob = resize_blob(input_blob, dest_shape=(ih, iw))
        input_blob = input_blob[np.newaxis, ...]

        ret_masks, ret_scores = gen_masks(self.net, input_blob, config, dest_shape=(oh, ow))

        encoded_masks = encode(ret_masks)
        reserved = np.ones((len(ret_masks)))
        for i in range(len(reserved)):
            if ret_scores[i] < self.threshold:
                reserved[i] = 0
                continue
            if reserved[i]:
                for j in range(i + 1, len(reserved)):
                    if reserved[j] and iou(encoded_masks[i], encoded_masks[j], [False]) > 0.5:
                        reserved[j] = 0

        results = []

        for _ in range(len(ret_masks)):
            if ret_scores[_] > self.threshold and reserved[_]:
                mask = ret_masks[_].copy()
                mask[mask == 1] = 0.3
                mask[mask == 0] = 1
                color = COLORS[_ % len(COLORS)]
                for k in range(3):
                    image[:,:,k] = image[:,:,k] * mask
                mask[mask == 1] = 0
                mask[mask > 0] = 0.7
                for k in range(3):
                    image[:,:,k] += mask * (color & 0xff)
                    color >>= 8;

                score = float(ret_scores[_])
                results.append({
                    'segmentation': encode(ret_masks[_]),
                    'score': score
                })


        frame_segmentation_obj = {
            'num_segmentations': len(results),
            'frame_segmentation_file_contents': image,
            'results': results
        }

        return frame_segmentation_obj



def main():
    parameters = '{"threshold": 0.85, "model": "fm-res39", "init_weights": "fm-res39_final.caffemodel"}'
    image_files = ['/home/paperspace/Pictures/fish_seg_data/fish_00_0.png']
    fga = FastMaskAlgorithm(parameters)
    
    images_path = "~/Pictures/fish_seg_data/"
    
    for file in os.listdir(images_path):
        if file.endswith(".png"):
            image_file = os.path.join(images_path, file)
    #for image_file in image_files:
            segmentation_obj = fga.generate_frame_segmentation(image_file)
            image_mat = segmentation_obj['frame_segmentation_file_contents']
            print image_mat
            cv2.imwrite('/tmp/segmented_image.png', image_mat)


if __name__ == '__main__':
    main()





            