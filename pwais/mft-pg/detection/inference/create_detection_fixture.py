import os
import typing

import attr
import click
import mlflow

import pandas as pd

from mft_utils import misc as mft_misc
from mft_utils.bbox2d import BBox2D
from mft_utils.img_w_boxes import ImgWithBoxes



def df_add_static_col(df, colname, v, empty_v=None):
  import six
  if empty_v is None:
    if hasattr(v, 'items'):
      empty_v = {}
    elif isinstance(v, list):
      empty_v = []
    elif isinstance(v, six.string_types):
      empty_v = ''
    else:
      empty_v = v
  df.insert(0, colname, [v] + ([empty_v] * max(0, len(df) - 1)))


###############################################################################
## Core Detection

class DarknetDetector(object):
  """Uses Darknet for inference"""
  
  def __init__(
          self, 
          config_path='yolov3-fish.cfg',
          weight_path='/outer_root/data8tb/pwais/yolo_fish_second/yolov3-fish_last.weights',
          meta_path='fish.data'): # Needed for class names
    
    # This import (typically) comes from /opt/darknet in the
    # dockerized environment
    import darknet
    self._darknet = darknet

    ## Based on darknet.performDetect()
    self._net = darknet.load_net_custom(
                    config_path.encode("ascii"),
                    weight_path.encode("ascii"),
                    0, 1)  # batch size = 1
    self._meta = darknet.load_meta(meta_path.encode("ascii"))

  def detect(
        self,
        img_path,
        confidence_thresh=0.25):
    
    import time
    start = time.time()
    detections = self._darknet.detect(
                          self._net,
                          self._meta,
                          img_path.encode("ascii"),
                          confidence_thresh)
    latency_sec = time.time() - start
    bboxes = []
    for detection in detections:
      pred_class = detection[0]
      confidence = detection[1]
      bounds = detection[2]
      # h, w = img.shape[:2]
      # x = shape[1]
      # xExtent = int(x * bounds[2] / 100)
      # y = shape[0]
      # yExtent = int(y * bounds[3] / 100)
      
      # Sometimes weekly-trained models spit out invalid boxes
      has_invalid = any(
        (b in (float('inf'), float('-inf'), float('nan')))
        for b in bounds)
      if has_invalid:
        continue

      yExtent = int(bounds[3])
      xEntent = int(bounds[2])
      # Coordinates are around the center
      xCoord = int(bounds[0] - bounds[2]/2)
      yCoord = int(bounds[1] - bounds[3]/2)

      bboxes.append(
        BBox2D(
          category_name=pred_class,
          x=xCoord,
          y=yCoord,
          width=xEntent,
          height=yExtent,
          score=confidence))
    
    return ImgWithBoxes(
              img_path=img_path,
              bboxes=bboxes,
              latency_sec=latency_sec)


class YoloAVTTRTDetector(object):
  """Wraps a TensorRT engine instance and provides basic inference
  functionality.
  """
  
  def __init__(
        self, trt_engine_path, input_hw, num_categories, category_id_to_name):
    self._trt_yolo = None
    self._engine_load_time_sec = -1.
    self._input_hw = (input_hw[0], input_hw[1])
    self._category_id_to_name = category_id_to_name

    import time

    assert os.path.exists('/opt/tensorrt_demos'), \
      "This runner designed to work with https://github.com/jkjung-avt/tensorrt_demos"

    import pycuda.autoinit  # This is needed for initializing CUDA driver
    from utils.yolo_with_plugins import TrtYOLO
      # From /opt/tensorrt_demos/utils/yolo_with_plugins.py

    class MyTrtYOLO(TrtYOLO):
      def _load_engine(self):
        import tensorrt as trt
        with open(trt_engine_path, 'rb') as f:
          with trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
      
      # def detect(self, *args, **kwargs):
      #   import pdb; pdb.set_trace()
      #   return super(MyTrtYOLO, self).detect(*args, **kwargs)
    
    start = time.time()
    mft_misc.log.info('Loading TRT engine %s' % trt_engine_path)
    self._trt_yolo = MyTrtYOLO(
        '', # ignored in our subclass
        self._input_hw,
        category_num=num_categories)
    self._engine_load_time_sec = time.time() - start
    mft_misc.log.info('Loaded TRT engine in %s' % self._engine_load_time_sec)

  @property
  def engine_load_time_sec(self):
    return self._engine_load_time_sec
  
  def detect(
        self,
        img_path='',
        img=None,
        auto_resize=True,
        confidence_thresh=0.1):
    
    import time

    if not img:
      assert os.path.exists(img_path)
      
      import imageio
      img = imageio.imread(img_path)
      
    resize_time_sec = -1.
    orig_input_hw = img.shape[:2]
    if img.shape[:2] != self._input_hw:
      # We will only resize the input if asked to do so
      assert auto_resize, (img.shape[:2], '!=', self._input_hw)

      import cv2
      h, w = self._input_hw
      start_resize = time.time()
      img = cv2.resize(img, (w, h))
      resize_time_sec = time.time() - start_resize

    assert self._trt_yolo is not None

    inf_start = time.time()
    boxes, confs, clss = self._trt_yolo.detect(img, confidence_thresh)
    latency_sec = time.time() - inf_start

    bboxes = []
    for bb, score, clazz in zip(boxes, confs, clss):
      # These are all in units of pixels! :)
      x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]

      if orig_input_hw != self._input_hw:
        scale_y = orig_input_hw[0] / self._input_hw[0]
        scale_x = orig_input_hw[1] / self._input_hw[1]
        x_min = x_min * scale_x
        y_min = y_min * scale_y
        x_max = x_max * scale_x
        y_max = y_max * scale_y

      clazz = int(clazz)

      bboxes.append(
        BBox2D(
          category_name=self._category_id_to_name[clazz],
          x=x_min,
          y=y_min,
          width=x_max - x_min + 1,
          height=y_max - y_min + 1,
          im_width=orig_input_hw[1],
          im_height=orig_input_hw[0],
          score=score))

    return ImgWithBoxes(
              img_path=img_path,
              bboxes=bboxes,
              latency_sec=latency_sec,
              extra={
                'confidence_thresh': str(confidence_thresh),
                'resize_time_sec': str(resize_time_sec),
              })



###############################################################################
## Detector Interface and Hooks

class DetectorRunner(object):
  def __call__(self, img_gts):
    
    img_dets = []
    for i, o in enumerate(img_gts):
      img_path = o.img_path
      result = self.detect_on_img_path(img_path)
      img_dets.append(result)

      if ((i+1) % 100) == 0:
        mft_misc.log.info('Detected %s of %s' % (i+1, len(img_gts)))

    mft_misc.log.info('Running COCO merics ...')
    from mft_utils import coco_detection_metrics as coco_metrics
    metrics = coco_metrics.get_coco_summary(img_gts, img_dets)
    mft_misc.log.info('... done')

    import copy
    import attr
    rows = []
    for gt, dt in zip(img_gts, img_dets):
      # Record ground truth used for evaluation; also helps viz.
      dt.bboxes_alt = gt.bboxes
      # [copy.deepcopy(bb) for bb in gt.bboxes]
      # for bb in dt.bboxes_alt:
      #   # NB: the coco_merics above need to have the same category_name to
      #   # work, but we prefix the categories here for bete
      #   bb.category_name = 'GT: ' + bb.category_name

      img_metrics = metrics['image_id_to_stats'][dt.img_path]
      dt.extra.update(
        ('coco_metrics_' + k, str(v))
        for k, v in img_metrics.items())

      row = attr.asdict(dt, recurse=False)
      row.update(
        ('img_coco_metrics_' + k, v)
        for k, v in img_metrics.items())
      rows.append(row)

    df = pd.DataFrame(rows)
    SKIP_GLOBAL_METRIC_KEYS = (
      'image_id_to_stats',
    )
    for k in metrics.keys():
      if k in SKIP_GLOBAL_METRIC_KEYS:
        continue
      v = metrics[k]
      df_add_static_col(df, 'coco_' + k, v)
    
    df_add_static_col(df, 'detector_runner', self.get_runner_name())
    df_add_static_col(df, 'detector_info', self.get_detector_info())
    # df.insert(0, 'detector_runner', self.get_runner_name())
    # df.insert(0, 'detector_info', self.get_detector_info())
    return df
  
  @classmethod
  def get_runner_name(cls):
    return str(cls.__name__)

  def get_detector_info(self):
    return {'base_runner': 'base_runner'}

  def detect_on_img_path(self, img_path):
    return ImgWithBoxes(img_path=img_path)


class DarknetRunner(DetectorRunner):
  def __init__(self, artifact_dir):
    config_path = os.path.join(artifact_dir, 'yolov3.cfg')
    weights_path = os.path.join(artifact_dir, 'yolov3_final.weights')

    # Make a fake meta file that just references the names file we need
    names_path = os.path.join(artifact_dir, 'names.names')

    import tempfile
    meta_path = tempfile.NamedTemporaryFile(suffix='.darknet.meta').name
    with open(meta_path, 'w') as f:
      f.write("names=" + names_path + '\n')
    
    self._detector = DarknetDetector(
                      config_path=config_path,
                      weight_path=weights_path,
                      meta_path=meta_path)
    
    w, h = mft_misc.darknet_get_yolo_input_wh(config_path)
    self._dector_info = {
      'config_path': config_path,
      'weights_path': weights_path,
      'input_w': w,
      'input_h': h,
    }
    
  def detect_on_img_path(self, img_path):
    return self._detector.detect(img_path=img_path)
  
  def get_detector_info(self):
    return self._dector_info


class YoloTRTRunner(DetectorRunner):
  
  @classmethod
  def get_runner_name(cls):
    engine_name = mft_misc.cuda_get_device_trt_engine_name()
    return str(cls.__name__) + '.' + engine_name
  
  def __init__(self, artifact_dir):
    engine_name = mft_misc.cuda_get_device_trt_engine_name()
    trt_engine_path = os.path.join(
                        artifact_dir, 'yolov3.%s.trt' % engine_name)
    yolo_config_path = os.path.join(artifact_dir, 'yolov3.cfg')
    
    w, h = mft_misc.darknet_get_yolo_input_wh(yolo_config_path)
    category_num = mft_misc.darknet_get_yolo_category_num(yolo_config_path)

    yolo_names_path = os.path.join(artifact_dir, 'names.names')
    category_id_to_name = open(yolo_names_path, 'r').readlines()

    self._detector = YoloAVTTRTDetector(
                        trt_engine_path,
                        (h, w),
                        category_num,
                        category_id_to_name)
    
    self._engine_load_time_sec = self._detector.engine_load_time_sec
    mlflow.log_metric('trt_engine_load_time_sec', self._engine_load_time_sec)

    self._dector_info = {
      'trt_engine_name': engine_name,
      'config_path': yolo_config_path,
      'input_w': w,
      'input_h': h,
    }

  def detect_on_img_path(self, img_path):
    result = self._detector.detect(img_path)
    result.extra['trt_engine_load_time_sec'] = self._engine_load_time_sec
    return result
  
  def get_detector_info(self):
    return self._dector_info


def create_runner_from_artifacts(artifact_dir):
  engine_name = mft_misc.cuda_get_device_trt_engine_name()
  if os.path.exists(os.path.join(artifact_dir, 'yolov3.%s.trt' % engine_name)):
    return YoloTRTRunner(artifact_dir)
  elif os.path.exists(os.path.join(artifact_dir, 'yolov3_final.weights')):
    return DarknetRunner(artifact_dir)
  else:
    raise ValueError("Could not resolve detector for %s" % artifact_dir)



@click.command(help="Run a detector and save detections as a Pandas Dataframe")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--detect_on_dataset", default="gopro1_fish_test",
  help="Create a detection fixture using this input")
@click.option("--detect_limit", default=-1,
  help="For testing, only run detection on this many samples")
@click.option("--save_to", default="", 
  help="Leave blank to write to a tempfile & log via mlflow")
@click.option("--gpu_id", default=-1,
  help="Use this GPU (default: unrestricted)")
def create_detection_fixture(
      use_model_run_id,
      use_model_artifact_dir,
      detect_on_dataset,
      detect_limit,
      save_to,
      gpu_id):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')
  
  assert use_model_artifact_dir, "Need some model artifacts to run a model"


  from mft_utils import gopro_data
  from mft_utils import akpd_data

  DATASET_NAME_TO_FACTORY = {}
  DATASET_NAME_TO_FACTORY.update(gopro_data.DATASET_NAME_TO_ITER_FACTORY)
  DATASET_NAME_TO_FACTORY.update(akpd_data.DATASET_NAME_TO_ITER_FACTORY)
  assert detect_on_dataset in DATASET_NAME_TO_FACTORY, \
    "Requested %s but only have %s" % (
      detect_on_dataset, DATASET_NAME_TO_FACTORY.keys())


  if gpu_id > 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


  run_id = use_model_run_id or None
  with mlflow.start_run(run_id=run_id) as mlrun:
    mlflow.log_param('parent_run_id', use_model_run_id)

    dfactory = DATASET_NAME_TO_FACTORY[detect_on_dataset]
    iter_img_gts = dfactory()

    if detect_limit >= 0:
      import itertools
      iter_img_gts = itertools.islice(iter_img_gts, detect_limit)
    img_gts = list(iter_img_gts)

    detector_runner = create_runner_from_artifacts(use_model_artifact_dir)

    import time
    start = time.time()
    df = detector_runner(img_gts)
    d_time = time.time() - start
    
    df_add_static_col(df, 'dataset', detect_on_dataset)
    df_add_static_col(df, 'model_artifact_dir', use_model_artifact_dir)
    df_add_static_col(df, 'model_run_id', use_model_run_id)
    # df.insert(0, 'dataset', detect_on_dataset)
    # df.insert(0, 'model_artifact_dir', use_model_artifact_dir)
    # df.insert(0, 'model_run_id', use_model_run_id)

    mlflow.log_metric('mean_latency_ms', 1e3 * df['latency_sec'].mean())

    mlflow.log_metric('total_detection_time_sec', d_time)

    if save_to:
      df.to_pickle(save_to)
    else:
      import tempfile
      fname = detector_runner.get_runner_name() + '.detections_df.pkl'
      save_to = os.path.join(tempfile.gettempdir(), fname)
      df.to_pickle(save_to)
      mlflow.log_artifact(save_to)
    mft_misc.log.info('Saved %s' % save_to)


if __name__ == "__main__":
  create_detection_fixture()
