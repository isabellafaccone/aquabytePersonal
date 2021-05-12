#!/usr/bin/env python3
import os
import sys
import json
import cv2
import pandas as pd
import numpy as np
import torch
import time
from torch2trt import torch2trt
from pathlib import Path
from datetime import datetime


sys.path.append('../biomass')

from akpd.utils import url_to_image
from image_quality.skip_classifierTRT import SkipPredictor

def benchmark(data_path, model_path, params_path, rows=-1):
    for path in [data_path, model_path, params_path]:
        if not path.exists():
            raise Exception(f'File not found {str(path)}')
            
    print('Loading test data...')
    test_data =  pd.read_pickle(str(data_path))
    l = len(test_data) if rows < 0 else rows
    
    print('Buillding infer engines...')
    skip_pred = SkipPredictor(model_path, params_path)
    
    i = 0
    print(f'Benchmarking {l} image-pairs...')
    samples=[]
    start_t = datetime.now()
    for ix,row in test_data.iterrows():
        if i % 10 == 0:
            print(f'[{datetime.now()}] Row {i} of {l}')

        state = row.state
        l_url = row.left_crop_url if not pd.isnull(row.left_crop_url) else ''
        r_url = row.right_crop_url if not pd.isnull(row.right_crop_url) else ''
        
        if l_url:
            image = url_to_image(l_url)
            samples.append(skip_pred.predict(image,ix,'l',state))
        if r_url:
            image = url_to_image(r_url)
            samples.append(skip_pred.predict(image,ix,'r',state))
        i += 1
        if i >= l:
            break
   
    t = datetime.now() - start_t
    
    return samples, t
    