import sys
import json
import cv2
import copy
import pandas as pd
import numpy as np
import pycuda.driver as cuda
import cudasift as cfs
from datetime import datetime,timedelta

from pathlib import Path

sys.path.append('../../biomass')

from akpd.utils import url_to_image, image_resize, enhance

from fishid import fishid
from fishid import util

def pykp2cvkeypoint(df):
    return [
        cv2.KeyPoint(
            x         = p.xpos,
            y         = p.ypos,
            _size     = p.scale,
            _angle    = p.orientation,
            _response = 0,
            _octave   = 0,
            _class_id = -1,
        )for _, p in df.iterrows()
    ]

def pysiftimage (image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image, dtype=np.float32)
    return image

def pysiftkps(image):
    image = pysiftimage(image)
    sift_data = cfs.PySiftData(20000)
    t = datetime.now()
    cfs.ExtractKeypoints(image, sift_data,numOctaves=5,thresh=1,upScale=False)
    data = sift_data.to_data_frame()
    t = (datetime.now() - t).total_seconds()
    kp = pykp2cvkeypoint(data[0])
    d = copy.deepcopy(data[1])
    sift_data.__deallocate__()
    return (kp,d),t

def cudamatch(image1,image2, ix, ux, tag_1, tag_2):
    data1, t1 = pysiftkps(image1)
    data2, t2 = pysiftkps(image2)
    t3 = datetime.now()
    inliers, is_match = fishid.find_matches(data1, data2, True)
    t3 = (datetime.now() - t3).total_seconds()
    return {
        'ix':ix,
        'ux':ux,
        'type':'CUDA_SIFT',
        'is_match':is_match,
        'gt_match':(tag_1==tag_2),
        'tag_i':tag_1,
        'tag_u':tag_2,
        'inliers':inliers,
        'kp_times':[t1,t2],
        'match_time':t3
    }

def cvsiftkps(image):
    t = datetime.now()
    data = fishid.get_kp_desc(image)
    t = (datetime.now() - t).total_seconds()
    return data, t

def cvmatch(image1,image2, ix, ux, tag_1, tag_2):
    data1, t1 = cvsiftkps(image1)
    data2, t2 = cvsiftkps(image2)
    t3 = datetime.now()
    inliers, is_match = fishid.find_matches(data1, data2)
    t3 = (datetime.now() - t3).total_seconds()
    return {
        'ix':ix,
        'ux':ux,
        'type':'CV_SIFT',
        'is_match':is_match,
        'gt_match':(tag_1==tag_2),
        'tag_i':tag_1,
        'tag_u':tag_2,
        'inliers':inliers,
        'kp_times':[t1,t2],
        'match_time':t3
    } 


def benchmark(data):
    cuda_sift = []
    cv_sift = []
    l = len(data)
    i = 0
    for ix, x in data.iterrows():
        print(f'[{str(datetime.now())}] Matching {i} of {l}')
        #a = url_to_image(x.left_image_url)
        a = url_to_image(x.right_image_url)
        a = fishid.enhance(a)
        tag_a = x['tag']
        for ux, y in data.iterrows():
            if ux == ix: continue
            if y.direction != x.direction: continue
            if len([w for w in cuda_sift if (w['ix'] == ux)]) > 0: continue
            if len([w for w in cv_sift if (w['ix'] == ux)]) > 0: continue
            tag_b = y['tag']
            b = url_to_image(y.left_image_url)
            #b = url_to_image(y.right_image_url)
            b = fishid.enhance(b)
            cuda_sift.append(cudamatch(a, b, ix,ux,tag_a,tag_b))
            cv_sift.append(cvmatch(a, b, ix, ux,tag_a,tag_b))
        i += 1
        
    return cuda_sift, cv_sift
                     
        