#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np

from datetime import datetime,timedelta

from .flags import Flags
from .akpd import Akpd
from .akpd_trt import AkpdTRT
from .keypoints import Keypoints, KP
from .utils import delta_frame

def benchmark(data_path, tf_model_path, trt_model_path, config_path, rows=-1):
    for path in [data_path, tf_model_path, trt_model_path, config_path]:
        if not path.exists():
            raise Exception(f'File not found {str(path)}')
    
    config = json.load(config_path.open('r'))
    FLAGS = Flags(config,str(tf_model_path))

    test_data =  pd.read_pickle(str(data_path))
    l = len(test_data) if rows < 0 else rows

    tf_run=[]
    kps_tf = []
    tf_time = None

    print('Creating TF Engine...')
    with Akpd(tf_model_path, config_path) as engine:
        assert engine, 'No TF Engine created'
        kp_tf = Keypoints(engine,FLAGS)
        i = 0
        print('Running TF infer...')
        start_t = datetime.now()
        for ix, row in test_data.iterrows():
            #if i % 10 == 0 :
            print(f'AKPD_TF {i} of {l}')
            kp, run_times = kp_tf.process(row)
            kps_tf.append(kp)
            tf_run.extend(run_times)
            i += 1
            if i >= rows and rows>=0:
                print(f'AKPD_TF {i} of {l} Done.')
                break
        tf_time = datetime.now() - start_t

    trt_fp32_run=[]
    kps_trt_fp32 = []
    trt_fp32_time = None
    print('Creating TRT FP32 Engine...')
    with AkpdTRT(trt_model_path, config_path) as engine:
        assert engine, 'No TRT Engine created'
        kp_trt = Keypoints(engine,FLAGS)
        i = 0
        print('Running TRT infer...')
        start_t = datetime.now()
        for ix, row in test_data.iterrows():
            #if i % 10 == 0:
            print(f'AKPD_TRT {i} of {l}')
            kp, run_times = kp_trt.process(row)
            kps_trt_fp32.append(kp)
            trt_fp32_run.extend(run_times)
            i += 1
            if i >= rows and rows>=0:
                print(f'AKPD_TRT {i} of {l} Done.')
                break
        trt_fp32_time = datetime.now() - start_t

    trt_fp16_run = []
    kps_trt_fp16 = []
    trt_fp16_time = None
    print('Creating TRT FP16 Engine...')
    with AkpdTRT(trt_model_path, config_path, fp16=True) as engine:
        assert engine, 'No TRT Engine created'
        kp_trt = Keypoints(engine,FLAGS)
        i = 0
        print('Running TRT FP16 infer...')
        start_t = datetime.now()
        for ix, row in test_data.iterrows():
            #if i % 50 == 0:
            print(f'AKPD_TRT {i} of {l}')
            kp, run_times = kp_trt.process(row)
            kps_trt_fp16.append(kp)
            trt_fp16_run.extend(run_times)
            i += 1
            if i >= rows and rows>=0:
                print(f'AKPD_TRT {i} of {l} Done.')
                break
        trt_fp16_time = datetime.now() - start_t
    
    delta_fp32 = delta_frame(kps_tf, kps_trt_fp32)
    d_fp32 = [x['point'] for d in delta_fp32 for x in d]

    delta_fp16 = delta_frame(kps_tf, kps_trt_fp16)
    d_fp16 = [x['point'] for d in delta_fp16 for x in d]
    
    return {
        'data_path': str(data_path),
        'rows': l,
        'tf':{
            'runtime':{
                'bench_time': str(tf_time),
                'avg':np.average(tf_run),
                'max':np.max(tf_run),
                'min':np.min(tf_run),
                'total': np.sum(tf_run)
            },
            'keypoints': kps_tf
        },
        'trt_fp32':{
            'runtime':{
                'bench_time': str(trt_fp32_time),
                'avg':np.average(trt_fp32_run),
                'max':np.max(trt_fp32_run),
                'min':np.min(trt_fp32_run),
                'total': np.sum(trt_fp32_run)
            },
            'keypoints': kps_trt_fp32,
            'diff': delta_fp32,
            'stat': {
                'samples': len(d_fp32), 
                'max': max(d_fp32),
                'min': min(d_fp32), 
                'avg': np.average(d_fp32),
                'std': np.std(d_fp32)
            }
        },
         'trt_fp16':{
            'runtime':{
                'bench_time': str(trt_fp16_time),
                'avg':np.average(trt_fp16_run),
                'max':np.max(trt_fp16_run),
                'min':np.min(trt_fp16_run),
                'total': np.sum(trt_fp16_run)
            },
            'keypoints': kps_trt_fp16,
            'diff': delta_fp16,
            'stat': {
                'samples': len(d_fp16), 
                'max': max(d_fp16),
                'min': min(d_fp16), 
                'avg': np.average(d_fp16),
                'std': np.std(d_fp16)
            }
        }
    }
