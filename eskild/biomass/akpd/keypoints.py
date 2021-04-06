#!/usr/bin/env python3

import numpy as np
import cv2

from .utils import url_to_image, image_resize, enhance
from .flags import Flags

KP = ["TAIL_NOTCH", "ADIPOSE_FIN", "UPPER_LIP", "ANAL_FIN", "PELVIC_FIN", "EYE", "PECTORAL_FIN", "DORSAL_FIN"]

class Keypoints(object):
    def __init__(self, akpd, flags):
        self._akpd = akpd
        self.FLAGS = flags
    
    @staticmethod
    def _get_keypoint(hm, kp, shape):
        hm = cv2.resize(hm[..., kp], shape)
        ii = np.unravel_index(np.argsort(hm.ravel())[-10000:], hm.shape)
        x = np.sum(np.exp(hm[ii]) * ii[1]) / np.sum(np.exp(hm[ii]))
        y = np.sum(np.exp(hm[ii]) * ii[0]) / np.sum(np.exp(hm[ii]))
        x = int(np.rint(x))
        y = int(np.rint(y))
        score = hm[y, x]
        hm_avg = np.mean(hm[ii])
        hm_max = hm.max()
        return {
            'point':[x,y],
            'score': score,
            'avg': hm_avg,
            'max': hm_max,
            'kp': KP[kp]
        }
        
    def process(self, row):
        imageL_url = row['left_image_url']
        imageR_url = row['right_image_url']

        imageL = url_to_image(imageL_url)
        imageR = url_to_image(imageR_url)

        imageL = enhance(imageL)
        imageR = enhance(imageR)

        heightL, widthL, _ = imageL.shape
        imageL = image_resize(imageL, self.FLAGS)

        heightR, widthR, _ = imageR.shape
        imageR = image_resize(imageR, self.FLAGS)

        hmL,hmR = self._akpd.process(imageL,imageR)

        kp_l = []
        kp_r = []

        for c, kp in enumerate(KP):
            kp_l.append(self._get_keypoint(hmL,c,(widthL, heightL)))
            kp_r.append(self._get_keypoint(hmR,c,(widthR, heightR)))
            
        return {
            'left_image':{
                'url': imageL_url,
                'kp': kp_l
            },
            'right_image':{
                'url': imageR_url,
                'kp': kp_r
            }
        }
