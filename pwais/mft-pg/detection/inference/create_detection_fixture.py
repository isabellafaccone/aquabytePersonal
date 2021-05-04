import os

import attr
import click
import mlflow

import pandas as pd

from mft_utils import misc as mft_misc
from mft_utils.bbox2d import BBox2D


@attr.s(slots=True, eq=True)
class ImgWithBoxes(object):

  img_path = attr.ib(default="")
  """img_path: Path to the image"""

  bboxes = attr.ib(default=[])
  """bboxes: A list of `BBox2D` instances"""


###############################################################################
## Datasets

def iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=[]):

  # NB: adapted from convert_to_yolo_format() in mft-pg
  # Some day mebbe refactor to join them...

  import csv

  with open(in_csv_path, newline='') as f:
    rows = list(csv.DictReader(f))

  print('Read %s rows from %s' % (len(rows), in_csv_path))

  # Read and cache the image dims only once; they're from a video so all
  # the same
  h = None
  w = None

  for row in rows:

    img_fname = os.path.basename(row['image_f'])
    img_path = os.path.join(imgs_basedir, img_fname)

    if (h, w) == (None, None):
      import imageio

      img = imageio.imread(img_path)
      h, w = img.shape[:2]
      
      print("Images have dimensions width %s height %s" % (w, h))

    # LOL those annotations aren't JSONs, they're string-ified python dicts :(
    import ast
    annos_raw = ast.literal_eval(row['annotation'])
    bboxes = []
    for anno in annos_raw['annotations']:
      if only_classes and anno['category'] not in only_classes:
        continue

      if 'xCrop' not in anno:
        print('bad anno %s' % (row,))
        continue

      x_pixels = anno['xCrop']
      y_pixels = anno['yCrop']
      w_pixels = anno['width']
      h_pixels = anno['height']

      bboxes.append(
        BBox2D(
          category_name=anno['category'],
          x=x_pixels,
          y=y_pixels,
          width=w_pixels,
          height=h_pixels,
          im_width=w,
          im_height=h))

    yield ImgWithBoxes(img_path=img_path, bboxes=bboxes)


DATASET_NAME_TO_ITER_FACTORY = {
  'gopro1_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=[])),
  'gopro1_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=[])),
  'gopro1_fish_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=['FISH'])),
  'gopro1_fish_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=['FISH'])),
  'gopro1_head_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=['HEAD'])),
  'gopro1_head_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=['HEAD'])),
}


###############################################################################
## Detection Runners

def run_yolo_detect(
      config_path='yolov3-fish.cfg',
      weight_path='/outer_root/data8tb/pwais/yolo_fish_second/yolov3-fish_last.weights',
      meta_path='fish.data', # Needed for class names
      img_paths=[]):
  
  # This import (typically) comes from /opt/darknet in the
  # dockerized environment
  import darknet
  
  import time

  ## Based on darknet.performDetect()

  net = darknet.load_net_custom(
                  config_path.encode("ascii"),
                  weight_path.encode("ascii"),
                  0, 1)  # batch size = 1
  meta = darknet.load_meta(meta_path.encode("ascii"))
  thresh = 0.25
  
  # TODO: figure out why darknet needs classes=1 for training but classes=2 for detect

  rows = []
  for i, path in enumerate(img_paths):
    start = time.time()
    detections = darknet.detect(net, meta, path.encode("ascii"), thresh)
    latency_ms = 1e-3 * (time.time() - start)
    boxes = []
    for detection in detections:
      pred_class = detection[0]
      confidence = detection[1]
      bounds = detection[2]
      # h, w = img.shape[:2]
      # x = shape[1]
      # xExtent = int(x * bounds[2] / 100)
      # y = shape[0]
      # yExtent = int(y * bounds[3] / 100)
      yExtent = int(bounds[3])
      xEntent = int(bounds[2])
      # Coordinates are around the center
      xCoord = int(bounds[0] - bounds[2]/2)
      yCoord = int(bounds[1] - bounds[3]/2)

      boxes.append(
        BBox2D(
          category_name=pred_class,
          x=xCoord,
          y=yCoord,
          width=xEntent,
          height=yExtent,
          score=confidence))
    rows.append({
      'img_path': path,
      'boxes': boxes,
      'latency_ms': latency_ms,
    })

    if (i % 100) == 0:
      print('detected', i+1, ' of ', len(img_paths))

  df = pd.DataFrame(rows)

  print("df['latency_ms'].mean()", df['latency_ms'].mean())

  return df

class DetectorRunner(object):
  def __call__(self, iter_img_gts):
    return pd.DataFrame([])

class DarknetRunner(DetectorRunner):
  def __init__(self, artifact_dir):
    self._artifact_dir = artifact_dir

  def __call__(self, iter_img_gts):
    config_path = os.path.join(self._artifact_dir, 'yolov3.cfg')
    weight_path = os.path.join(self._artifact_dir, 'yolov3_final.weights')

    # Make a fake meta file that just references the names file we need
    names_path = os.path.join(self._artifact_dir, 'names.names')

    import tempfile
    meta_path = tempfile.NamedTemporaryFile(suffix='.darknet.meta').name
    with open(meta_path, 'w') as f:
      f.write("names=" + names_path + '\n')
    
    df = run_yolo_detect(
          config_path=config_path,
          weight_path=weight_path,
          meta_path=meta_path,
          img_paths=[o.img_path for o in iter_img_gts])
    return df

def create_runner_from_artifacts(artifact_dir):
  assert False, (os.path.join(artifact_dir, 'yolov3_final.weights'), os.path.exists(os.path.join(artifact_dir, 'yolov3_final.weights')))
  if os.path.exists(os.path.join(artifact_dir, 'yolov3_final.weights')):
    return DarknetRunner(artifact_dir)
  else:
    raise ValueError("Could not resolve detector for %s" % artifact_dir)



# todo lets use this https://github.com/rafaelpadilla/review_object_detection_metrics







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
def create_detection_fixture(
      use_model_run_id,
      use_model_artifact_dir,
      detect_on_dataset,
      detect_limit,
      save_to):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
  
  assert use_model_artifact_dir, "Need some model artifacts to run a model"

  assert detect_on_dataset in DATASET_NAME_TO_ITER_FACTORY, \
    "Requested %s but only have %s" % (
      detect_on_dataset, DATASET_NAME_TO_ITER_FACTORY.keys())

  iter_factory = DATASET_NAME_TO_ITER_FACTORY[detect_on_dataset]
  iter_img_gts = iter_factory()

  if detect_limit >= 0:
    import itertools
    iter_img_gts = itertools.islice(iter_img_gts, detect_limit)

  detector_runner = create_runner_from_artifacts(use_model_artifact_dir)
  
  with mlflow.start_run() as mlrun:
    import time
    print('running detector ...')
    start = time.time()
    df = detector_runner(iter_img_gts)
    d_time = time.time() - start
    print('done running in', d_time)
    mlflow.log_param('total_detection_time_sec', d_time)

    if save_to:
      df.to_csv(save_to)
    else:
      import tempfile
      save_to = os.path.join(tempfile.gettempdir(), mlrun.info.run_id)
      df.to_csv(save_to)
      mlflow.log_artifact(save_to)

if __name__ == "__main__":
  create_detection_fixture()