#!/usr/bin/env python3
import json
import pandas as pd

from datetime import datetime,timedelta

from .flags import Flags
from .akpd import Akpd
from .akpd_trt import AkpdTRT
from .keypoints import Keypoints, KP

def benchmark(data_path, tf_model_path, trt_model_path, config_path, rows=-1, tf_bench=None):
    for path in [data_path, tf_model_path, trt_model_path, config_path]:
        if not path.exists():
            raise Exception(f'File not found {str(path)}')
    
    config = json.load(config_path.open('r'))
    FLAGS = Flags(config,str(tf_model_path))

    test_data =  pd.read_pickle(str(data_path))
    l = len(test_data) if rows < 0 else rows

    kps_tf = []
    tf_time = None
    if not tf_bench:
        print('Creating TF Engine...')
        with Akpd(tf_model_path, config_path) as engine:
            assert engine, 'No TF Engine created'
            kp_tf = Keypoints(engine,FLAGS)
            i = 0
            print('Running TF infer...')
            start_t = datetime.now()
            for ix, row in test_data.iterrows():
                if i % 50 == 0 :
                    print(f'AKPD_TF {i} of {l}')
                kps_tf.append(kp_tf.process(row))
                i += 1
                if i >= rows and rows>=0:
                    print(f'AKPD_TF {i} of {l} Done.')
                    break
            tf_time = datetime.now() - start_t
    else:
        kps_tf = tf_bench

    kps_trt_fp32 = []
    trt_time = None
    print('Creating TRT FP32 Engine...')
    with AkpdTRT(trt_model_path, config_path) as engine:
        assert engine, 'No TRT Engine created'
        kp_trt = Keypoints(engine,FLAGS)
        i = 0
        print('Running TRT infer...')
        start_t = datetime.now()
        for ix, row in test_data.iterrows():
            if i % 50 == 0:
                print(f'AKPD_TRT {i} of {l}')
            kps_trt_fp32.append(kp_trt.process(row))
            i += 1
            if i >= rows and rows>=0:
                print(f'AKPD_TRT {i} of {l} Done.')
                break
        trt_time = datetime.now() - start_t
        
    kps_trt_fp16 = []
    trt_time = None
    print('Creating TRT FP16 Engine...')
    with AkpdTRT(trt_model_path, config_path, fp16=True) as engine:
        assert engine, 'No TRT Engine created'
        kp_trt = Keypoints(engine,FLAGS)
        i = 0
        print('Running TRT infer...')
        start_t = datetime.now()
        for ix, row in test_data.iterrows():
            if i % 50 == 0:
                print(f'AKPD_TRT {i} of {l}')
            kps_trt_fp16.append(kp_trt.process(row))
            i += 1
            if i >= rows and rows>=0:
                print(f'AKPD_TRT {i} of {l} Done.')
                break
        trt_time = datetime.now() - start_t

    return {
        'data_path': str(data_path),
        'rows': l,
        'tf':{
            'runtime':tf_time,
            'keypoints': kps_tf
        },
        'trt_fp32':{
            'runtime':trt_time,
            'keypoints': kps_trt_fp32
        },
         'trt_fp16':{
            'runtime':trt_time,
            'keypoints': kps_trt_fp16
        }
    }