import os

import attr
import click
import mlflow

from mft_pg.bbox2d import BBox2D


@attr.s(slots=True, eq=True)
class ImgWithBoxes(object):

  img_path = attr.ib(default="")
  """img_path: Path to the image"""

  bboxes = attr.ib(default=[])
  """bboxes: A list of `BBox2D` instances"""


def iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/'):

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


def run_yolo_detect(
      config_path='yolov3-fish.cfg',
      weight_path='/outer_root/data8tb/pwais/yolo_fish_second/yolov3-fish_last.weights',
      meta_path='fish.data',
      class_to_id={b'idk': 0, b'FISH': 1},
      img_paths=[]):
  
  import darknet
  import imageio
  import time

  ## Based on darknet.performDetect()

  net = darknet.load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
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
      label = detection[0]
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

      boxes.append({
        'label': label,
        'class_id': class_to_id[label],
        'confidence': confidence,
        
        # Use a format friendly to motrackers package
        'x_min_pixels': xCoord,
        'y_min_pixels': yCoord,
        'width_pixels': xEntent,
        'height_pixels': yExtent,
      })
    rows.append({
      'img_path': path,
      'boxes': boxes,
      'latency_ms': latency_ms,
    })

    if (i % 100) == 0:
      print('detected', i)

  import pandas as pd
  df = pd.DataFrame(rows)

  print("df['latency_ms'].mean()", df['latency_ms'].mean())

  return df

@click.command(help="Run a detector and save detections as a Pandas Dataframe")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--detect_on_dataset", default="gopro1_fish",
  help="Create a detection fixture using this input")
@click.option("--output", default="", 
  help="Leave blank to write to a tempfile & log via mlflow")
def create_detection_fixture(
      use_model_run_id,
      use_model_artifact_dir,
      detect_on_dataset,
      output):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
  
  assert use_model_artifact_dir, "Need some model artifacts to run a model"



    

if __name__ == "__main__":
  create_detection_fixture()