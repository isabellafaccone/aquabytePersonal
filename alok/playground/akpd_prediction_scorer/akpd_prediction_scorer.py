import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import load_model


class AKPDPredictionScorer(object):
    
    def __init__(self, model_f, body_parts):
        self.model = load_model(model_f)
        self.body_parts = sorted(body_parts)

    def _get_left_right_keypoints(self, keypoints):
        left_keypoints, right_keypoints = {}, {}
        for item in keypoints['leftCrop']:
            left_keypoints[item['keypointType']] = np.array([item['xFrame'], item['yFrame']])

        for item in keypoints['rightCrop']:
            right_keypoints[item['keypointType']] = np.array([item['xFrame'], item['yFrame']])

        return left_keypoints, right_keypoints

    
    def _rotate(self, point, angle, origin=(0, 0)):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy


    def _normalize_keypoints(self, keypoints, origin_bp='TAIL_NOTCH'):
        # translation
        for bp in body_parts:
            keypoints[bp] = keypoints[bp] - keypoints[origin_bp]
            keypoints[bp][1] = -keypoints[bp][1]

        # rotation & compression
        angle = np.arctan(keypoints['UPPER_LIP'][1] / keypoints['UPPER_LIP'][0])
        for bp in body_parts:
            keypoints[bp] = self._rotate(keypoints[bp], -angle)
            keypoints[bp] = keypoints[bp] / np.linalg.norm(keypoints['UPPER_LIP'])

        return keypoints
    
    def _generate_one_side_score(self, coords):
        X = np.array([coords, ])
        X = np.swapaxes(X, 1, 2)
        return self.model.predict(X)
        


    def get_confidence_score(self, pred_keypoints):

        pred_left_keypoints, pred_right_keypoints = self._get_left_right_keypoints(pred_keypoints)
        pred_norm_left_keypoints = self._normalize_keypoints(pred_left_keypoints)
        pred_norm_right_keypoints = self._normalize_keypoints(pred_right_keypoints)

        coords_left, coords_right = [], []
        for bp in self.body_parts:
            coords_left.append(pred_norm_left_keypoints[bp])
            coords_right.append(pred_norm_right_keypoints[bp])
            
        left_score = self._generate_one_side_score(coords_left)[0][0]
        right_score = self._generate_one_side_score(coords_right)[0][0]
        return min(left_score, right_score)

def main():

    pred_keypoints = {"version": 2, "leftCrop": [{"xCrop": 58, "yCrop": 367, "xFrame": 382, "yFrame": 959, "keypointType": "UPPER_LIP"}, {"xCrop": 232, "yCrop": 345, "xFrame": 556, "yFrame": 937, "keypointType": "EYE"}, {"xCrop": 724, "yCrop": 70, "xFrame": 1048, "yFrame": 662, "keypointType": "DORSAL_FIN"}, {"xCrop": 1255, "yCrop": 150, "xFrame": 1579, "yFrame": 742, "keypointType": "ADIPOSE_FIN"}, {"xCrop": 1426, "yCrop": 209, "xFrame": 1750, "yFrame": 801, "keypointType": "UPPER_PRECAUDAL_PIT"}, {"xCrop": 1525, "yCrop": 275, "xFrame": 1849, "yFrame": 867, "keypointType": "HYPURAL_PLATE"}, {"xCrop": 1623, "yCrop": 283, "xFrame": 1947, "yFrame": 875, "keypointType": "TAIL_NOTCH"}, {"xCrop": 1430, "yCrop": 328, "xFrame": 1754, "yFrame": 920, "keypointType": "LOWER_PRECAUDAL_PIT"}, {"xCrop": 1187, "yCrop": 423, "xFrame": 1511, "yFrame": 1015, "keypointType": "ANAL_FIN"}, {"xCrop": 900, "yCrop": 484, "xFrame": 1224, "yFrame": 1076, "keypointType": "PELVIC_FIN"}, {"xCrop": 466, "yCrop": 462, "xFrame": 790, "yFrame": 1054, "keypointType": "PECTORAL_FIN"}], "rightCrop": [{"xCrop": 21, "yCrop": 392, "xFrame": 83, "yFrame": 961, "keypointType": "UPPER_LIP"}, {"xCrop": 185, "yCrop": 363, "xFrame": 247, "yFrame": 932, "keypointType": "EYE"}, {"xCrop": 708, "yCrop": 78, "xFrame": 770, "yFrame": 647, "keypointType": "DORSAL_FIN"}, {"xCrop": 1261, "yCrop": 171, "xFrame": 1323, "yFrame": 740, "keypointType": "ADIPOSE_FIN"}, {"xCrop": 1462, "yCrop": 228, "xFrame": 1524, "yFrame": 797, "keypointType": "UPPER_PRECAUDAL_PIT"}, {"xCrop": 1538, "yCrop": 294, "xFrame": 1600, "yFrame": 863, "keypointType": "HYPURAL_PLATE"}, {"xCrop": 1645, "yCrop": 302, "xFrame": 1707, "yFrame": 871, "keypointType": "TAIL_NOTCH"}, {"xCrop": 1445, "yCrop": 345, "xFrame": 1507, "yFrame": 914, "keypointType": "LOWER_PRECAUDAL_PIT"}, {"xCrop": 1198, "yCrop": 443, "xFrame": 1260, "yFrame": 1012, "keypointType": "ANAL_FIN"}, {"xCrop": 901, "yCrop": 523, "xFrame": 963, "yFrame": 1092, "keypointType": "PELVIC_FIN"}, {"xCrop": 414, "yCrop": 481, "xFrame": 476, "yFrame": 1050, "keypointType": "PECTORAL_FIN"}]}
    body_parts = sorted([
        'UPPER_LIP',
        'TAIL_NOTCH',
        'PECTORAL_FIN',
        'PELVIC_FIN',
        'ADIPOSE_FIN',
        'EYE',
        'DORSAL_FIN',
        'ANAL_FIN'
    ])

    f = './akpd_scorer_model.h5'
    aps = AKPDPredictionScorer(f, body_parts)
    aps.get_confidence_score(pred_keypoints)

if __name__ == '__main__':
    main()