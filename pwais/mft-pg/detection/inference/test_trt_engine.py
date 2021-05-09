# Based upon tensorrt_demos
# https://github.com/jkjung-avt/tensorrt_demos/blob/0016973315d1f3f6eaed70a5abd03d6309fe4730/trt_yolo.py#L1

import os

DEFAULT_IMG_FILE_PATHS = [
  os.path.join('/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/', fname)
  for fname in (
    'frame_2551.jpg',
    'frame_2552.jpg',
    'frame_2553.jpg',
    'frame_2554.jpg',
    'frame_2555.jpg',
    'frame_2556.jpg',
    'frame_2557.jpg',
    'frame_2558.jpg',
    'frame_2559.jpg',
    'frame_2560.jpg',
    'frame_2561.jpg',
    'frame_2562.jpg',
    'frame_2563.jpg',
    'frame_2564.jpg',
    'frame_2565.jpg',
    'frame_2566.jpg',
    'frame_2567.jpg',
  )
]

def run_trt_yolo(
      img_paths=DEFAULT_IMG_FILE_PATHS,
      trt_engine_path='/opt/mft-pg/detection/models/detection_models_s3/yolo_ragnarok_config_hack0/yolov3-416.trt',
      model_hw=(416, 416),
      model_num_categories=2):



  import time

  import cv2
  import imageio
  import numpy as np

  import pycuda.autoinit  # This is needed for initializing CUDA driver
  from utils.yolo_with_plugins import TrtYOLO

  class MyTrtYOLO(TrtYOLO):
    def _load_engine(self):
      import tensorrt as trt
      with open(trt_engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

  start = time.time()
  print('loading %s' % trt_engine_path)
  trt_yolo = MyTrtYOLO(
    '', # ignored in our subclass
    model_hw,
    category_num=model_num_categories)
  engine_load_time = time.time() - start
  print('loaded TRT engine in ', engine_load_time)
  for path in img_paths:
    img = imageio.imread(path)
    h, w = model_hw
    img = cv2.resize(img, (w, h))

    start = time.time()
    conf_th = 0.25
    boxes, confs, clss = trt_yolo.detect(img, conf_th)
    inf_time = time.time() - start

    print(path, inf_time)




  # from /opt/tensorrt_demos

if __name__ == '__main__':
  run_trt_yolo()
