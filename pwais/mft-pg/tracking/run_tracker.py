import os

import click
import mlflow

import pandas as pd

from mft_utils import misc as mft_misc
from mft_utils.bbox2d import BBox2D


# def run_tracker(
#      detections_df, debug_video_dest='tracks.mp4'):

#   import numpy as np
#   import imageio
#   from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
#   from motrackers.utils import draw_tracks

#   tracker = SORT(max_lost=3, trakcer_output_format='mot_challenge', iou_threshold=0.3)

#   writer = imageio.get_writer(debug_video_dest, fps=10)
#   detections_df.sort_values('img_path')
#   for idx, row in detections_df.iterrows():
#     img_path = row['img_path']
#     boxes = row['boxes']

#     bboxes = np.array([
#       [b['x_min_pixels'], b['y_min_pixels'], b['width_pixels'], b['height_pixels']]
#       for b in boxes
#     ])
#     confidences = np.array([b['confidence'] for b in boxes])
#     class_ids = np.array([b['class_id'] for b in boxes])

#     tracks = tracker.update(bboxes, confidences, class_ids)

#     debug_img = imageio.imread(img_path)
#     debug_img = _draw_bboxes(debug_img, bboxes, confidences, class_ids)

#     debug_img = draw_tracks(debug_img, tracks)

#     writer.append_data(debug_img)
#     print('tracked', idx, img_path)
#   writer.close()


def create_runner_from_artifacts(artifact_dir):
  pass
  # if os.path.exists(os.path.join(artifact_dir, 'yolov3.trt')):
  #   return YoloTRTRunner(artifact_dir)
  # elif os.path.exists(os.path.join(artifact_dir, 'yolov3_final.weights')):
  #   return DarknetRunner(artifact_dir)
  # else:
  #   raise ValueError("Could not resolve detector for %s" % artifact_dir)



# todo lets use this https://github.com/rafaelpadilla/review_object_detection_metrics







@click.command(help="Run a Tracker and save tracks as a Pandas Dataframe")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--tracker_algo", default="motrackers.SORT",
  help="Use this tracking algo")
@click.option("--save_debugs", default=True, 
  help="Save debug output as well as tracking results")
@click.option("--save_to", default="", 
  help="Leave blank to write to a tempdir & log via mlflow")
def run_tracker(
      use_model_run_id,
      use_model_artifact_dir,
      tracker_algo,
      save_debugs,
      save_to):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')
  
  assert use_model_artifact_dir, "Need some model artifacts to run a model"





  # assert detect_on_dataset in DATASET_NAME_TO_ITER_FACTORY, \
  #   "Requested %s but only have %s" % (
  #     detect_on_dataset, DATASET_NAME_TO_ITER_FACTORY.keys())

  # iter_factory = DATASET_NAME_TO_ITER_FACTORY[detect_on_dataset]
  # iter_img_gts = iter_factory()

  # if detect_limit >= 0:
  #   import itertools
  #   iter_img_gts = itertools.islice(iter_img_gts, detect_limit)

  # detector_runner = create_runner_from_artifacts(use_model_artifact_dir)
  
  # if gpu_id > 0:
  #   import os
  #   os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

  # with mlflow.start_run() as mlrun:
  #   mlflow.log_param('parent_run_id', use_model_run_id)

  #   import time
  #   start = time.time()
  #   df = detector_runner(iter_img_gts)
  #   d_time = time.time() - start
    
  #   mlflow.log_metric('mean_latency_ms', 1e3 * df['latency_sec'].mean())

  #   mlflow.log_metric('total_detection_time_sec', d_time)

  #   if save_to:
  #     df.to_pickle(save_to)
  #   else:
  #     import tempfile
  #     fname = str(mlrun.info.run_id) + '_detections_df.pkl'
  #     save_to = os.path.join(tempfile.gettempdir(), fname)
  #     df.to_pickle(save_to)
  #     mlflow.log_artifact(save_to)
  #   mft_misc.log.info('Saved %s' % save_to)


if __name__ == "__main__":
  run_tracker()
