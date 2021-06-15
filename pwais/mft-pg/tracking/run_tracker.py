import copy
import itertools
import os

import click
import mlflow

import pandas as pd

from mft_utils import df_util
from mft_utils import misc as mft_misc
from mft_utils import tracking
from mft_utils.img_w_boxes import ImgWithBoxes
from mft_utils.gopro_data import DATASET_NAME_TO_ITER_FACTORY

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

class TrackerRunnerBase(object):

  @classmethod
  def get_name(cls):
    return str(cls.__name__)
  
  def get_tracker_info(self):
    return {}

  def update_and_fill_tracks(self, imbb):
    return None

  def track_all(
        self,
        imbbs,
        ablate_input_to_fps=-1,
        debug_output_path='',
        debug_parallel=-1):

    ablation_mode = (ablate_input_to_fps >= 0)

    if ablation_mode:
      target_period_micros = int((1. / ablate_input_to_fps) * 1e6)
      last_tracker_update_micros = -target_period_micros
    else:
      target_period_micros = -1
      last_tracker_update_micros = -1
    
    imbbs_out = []
    last_bbs = []
    for i in range(len(imbbs)):
      imbb = copy.deepcopy(imbbs[i])
      if ablation_mode:
        imbb.bboxes_alt = copy.deepcopy(imbb.bboxes)
        imbb.extra['tracker_gt_is_ablation'] = 'True'
      else:
        imbb.extra['tracker_gt_is_ablation'] = 'False'

      should_update = (
        (not ablation_mode) or
        imbb.microstamp >= (last_tracker_update_micros + target_period_micros)
      )

      if should_update:
        self.update_and_fill_tracks(imbb)
        imbb.extra['did_run_tracker'] = 'True'
        if ablation_mode:
          last_bbs = copy.deepcopy(imbb.bboxes)
        last_tracker_update_micros = imbb.microstamp
      else:
        imbb.bboxes = copy.deepcopy(last_bbs)
        imbb.extra['did_run_tracker'] = 'False'
      
      imbbs_out.append(imbb)

      if ((i+1) % 100) == 0:
        mft_misc.log.info('Tracked %s of %s' % (i+1, len(imbbs)))

    df = df_util.to_obj_df(imbbs_out)
    df_util.df_add_static_col(
      df, 'tracker_name', self.get_name())
    df_util.df_add_static_col(
      df, 'tracker_info', self.get_tracker_info())
    df_util.df_add_static_col(
      df, 'tracker_ablate_input_to_fps', ablate_input_to_fps)

    # if debug_output_path:

    #   ## Render debug video
    #   debug_video_dest = os.path.join(debug_output_path, 'tracks_debug.mp4')
    #   tracking.write_debug_video(
    #     debug_video_dest, imbbs_out, parallel=debug_parallel)

    return df


class MOTrackerRunner(TrackerRunnerBase):
  def __init__(self, tracker_type):
    self._motracker = tracking.MOTrackersTracker(tracker_type=tracker_type)
    self._last_bb = []

  def get_tracker_info(self):
    return dict(
      tracker_type=self._motracker.tracker_type,
      tracker_kwargs=self._motracker.tracke_params,
    )
  
  def update_and_fill_tracks(self, imbb):
    res = self._motracker.update_and_fill_tracks(imbb)
    self._last_bb = copy.deepcopy(imbb.bboxes)
    return res



class TrackerRunner(object):

  def __init__(self):
    self._tracker = None

  


def create_runner_from_conf(tracker_conf):
  if tracker_conf.startswith('motrackers.'):
    tracker_conf = tracker_conf.replace('motrackers.', '')
    return MOTrackerRunner(tracker_conf)
  else:
    raise ValueError("Could not resolve tracker for %s" % tracker_conf)



# todo lets use this https://github.com/rafaelpadilla/review_object_detection_metrics







@click.command(help="Run a Tracker and save tracks as a Pandas Dataframe")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--use_dataset", default="",
  help="Induce a sequence of detections from this dataset")
@click.option("--use_detections_df", default="",
  help="Induce a sequence of detections from this Pickled dataframe")
@click.option("--tracker_conf", default="motrackers.SORT",
  help="Use this tracking config")
@click.option("--ablate_input_to_fps", default=-1,
  help="Ablate input to this Frames Per Second")
@click.option("--save_to", default="", 
  help="Leave blank to write to a tempdir & log via mlflow")
def run_tracker(
      use_model_run_id,
      use_model_artifact_dir,
      use_dataset,
      use_detections_df,
      tracker_conf,
      ablate_input_to_fps,
      save_to):
  
  
  img_boxes = None
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')
    save_to = save_to or use_model_artifact_dir
  
  if use_detections_df:
    detections_df = pd.read_pickle(use_detections_df)
    img_boxes = [
      ImgWithBoxes.from_dict(r)
      for r in detections_df.to_dict(orient='records')
    ]
    save_to = save_to or os.path.dirname(use_detections_df)

  if use_dataset:
    iter_factory = DATASET_NAME_TO_ITER_FACTORY[use_dataset]
    iter_img_gts = iter_factory()
    img_boxes = list(iter_img_gts)
    
    save_to = save_to or '/tmp/debug_' + use_dataset
    mft_misc.mkdir(save_to)

  assert img_boxes, "Need some image-boxes to run on"
  assert os.path.exists(save_to), "Need some directory to save output"

  runner = create_runner_from_conf(tracker_conf)
  df = runner.track_all(
                img_boxes,
                ablate_input_to_fps=ablate_input_to_fps,
                debug_output_path=save_to)
  
  df_util.df_add_static_col(
    df,
    'dataset',
    use_dataset or detections_df['dataset'][0])
  df_util.df_add_static_col(
    df,
    'model_artifact_dir',
    use_model_artifact_dir or (
      detections_df['model_artifact_dir'][0] if use_detections_df
      else 'anon'))
  df_util.df_add_static_col(
    df,
    'model_run_id',
    use_model_run_id or (
      detections_df['model_run_id'][0] if use_detections_df
      else 'anon'))

  obj_df = df_util.to_obj_df(df)
  
  df_dest_fname = 'tracks_df.pkl'
  if use_detections_df:
    df_dest_fname = os.path.basename(use_detections_df) + '.' + df_dest_fname
  elif use_dataset:
    df_dest_fname = use_dataset + '.' + df_dest_fname
  obj_df.to_pickle(os.path.join(save_to, df_dest_fname))




  # import numpy as np
  # import imageio
  # from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
  # from motrackers.utils import draw_tracks

  # tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)

  # writer = imageio.get_writer('tracks.mp4', fps=60)
  # TARGET_FPS = 60.
  # TARGET_PERIOD_MICROS = int((1. / TARGET_FPS) * 1e6)
  # last_tracker_update_micros = -TARGET_PERIOD_MICROS
  
  # tracks = []
  # bboxes = np.zeros((0, 4))
  # confidences = np.zeros((0, 1))
  # class_ids = np.zeros((0, 1))

  # class_id_to_name = sorted(set(itertools.chain.from_iterable(
  #                       (b.category_name for b in d.bboxes)
  #                       for d in img_boxes)))
  # class_name_to_id = dict((v, k) for k, v in enumerate(class_id_to_name))

  # for ib in img_boxes:
  #   img_path = ib.img_path
  #   boxes = ib.bboxes

  #   # bboxes = np.array([
  #   #   [b['x_min_pixels'], b['y_min_pixels'], b['width_pixels'], b['height_pixels']]
  #   #   for b in boxes
  #   # ])
    

  #   if ib.microstamp >= (last_tracker_update_micros + TARGET_PERIOD_MICROS):
  #     bboxes = np.array([
  #       [b.x, b.y, b.width, b.height]
  #       for b in boxes
  #     ])
  #     confidences = np.array([b.score for b in boxes])
  #     class_ids = np.array([class_name_to_id[b.category_name] for b in boxes])

  #     import time
  #     start = time.time()
  #     tracks = tracker.update(bboxes, confidences, class_ids)
  #     print('update time', time.time() - start)
  #     # print(bboxes - np.array([[t[2], t[3], t[4], t[5]] for t in tracks]))
  #     print('given but not tracked',
  #       set(tuple(b) for b in bboxes) - 
  #       set((t[2], t[3], t[4], t[5]) for t in tracks))
  #     print('tracked but not given',
  #       set((t[2], t[3], t[4], t[5]) for t in tracks) - 
  #       set(tuple(b) for b in bboxes))
      
  #     last_tracker_update_micros = ib.microstamp

  #   debug_img = imageio.imread(img_path)

  #   # def _draw_bboxes(image, bboxes, confidences, class_ids):
  #   #   """
  #   #   based upon motrackers.detector.draw_bboxes()

  #   #   Draw the bounding boxes about detected objects in the image.

  #   #   Parameters
  #   #   ----------
  #   #   image : numpy.ndarray
  #   #       Image or video frame.
  #   #   bboxes : numpy.ndarray
  #   #       Bounding boxes pixel coordinates as (xmin, ymin, width, height)
  #   #   confidences : numpy.ndarray
  #   #       Detection confidence or detection probability.
  #   #   class_ids : numpy.ndarray
  #   #       Array containing class ids (aka label ids) of each detected object.

  #   #   Returns
  #   #   -------
  #   #   numpy.ndarray : image with the bounding boxes drawn on it.
  #   #   """
  #   #   import cv2 as cv
  #   #   BBOX_COLORS = {0: (255, 255, 0), 1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
  #   #   for bb, conf, cid in zip(bboxes, confidences, class_ids):
  #   #       clr = [int(c) for c in BBOX_COLORS[cid]]
  #   #       cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
  #   #       label = "{}:{:.4f}".format(cid, conf)
  #   #       (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
  #   #       y_label = max(bb[1], label_height)
  #   #       cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
  #   #                     (255, 255, 255), cv.FILLED)
  #   #       cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
  #   #   return image

  #   for bbox in boxes:
  #     bbox.draw_in_image(debug_img)
  #   # debug_img = _draw_bboxes(debug_img, bboxes, confidences, class_ids)

  #   debug_img = draw_tracks(debug_img, tracks)

  #   writer.append_data(debug_img)
  #   print('tracked', img_path)
  # writer.close()

"""
cd /opt/ && \
  git clone https://github.com/adipandas/multi-object-tracker  && \
    cd multi-object-tracker && \
      git checkout 501605262a120f1ab47658f423fa180a6f36117c && \
        pip3 install -r requirements.txt && \
        pip3 install ipyfilechooser && \
        pip3 install -e .




deply demo ticket:
 ** step 1: create well-defined end-to-end script to run on a fixture of data.  for now just yolo + tracker
 ** bigger experiment for demo + data mining. run yolo detector + tracker on unlabeled gopro videos and measure:
    - raw distinct fish per minute (and over all M minutes of video)
    - *** ablate the feed down to N frames per second and repeat experiments
    - TX2 latency histogram of detector and tracker, maybe pre-loading N-seconds into RAM,
         other TX2 vitals
    - use Histogram With Examples reports to surface:
        * identify time segments of no fish and check WTF
        * identify time segments with lots of newly-spawned tracks and check WTF


 * tracklet scorer ticket:
    - part 1: translate AKPD keypoints to bboxen (and maybe pairs of bboxes) and test yolo on it.  if its good then:
    - devise a 'max distance scorer' for tracklets and do a Histogram with Examples study over the training data
    - if scores make sense, then run over all gopro videos to estimate good fish per hour.  do a 
          histogram with examples report to surface

 * accuracy vs latency ticket:
    - use the dump of N-thousand bbox examples to train & test yolos just as we did for gopro
    - add COCO metrics







 * ablation test...
    * add timestamp to imgbbox.
    * make a thing to ablate imgbbox sequence.
    * given input sequence, do ablations and output those tracks
       * output: MOTS comparison between full res seq and ablated
       * output: debug video of ablated tracking, showing intermediate frames?
    * we can start out just looking at "ground truth".  in the future want to
        plug in yolo.


where is this going?
 * given a set of bboxes, just output a tracking fixture with a debug
 * ... just output above, but ablate frame rate
 * compare tracking fixtures

-- the ablation thing probably gonna be a "one and done" to estimate robustness
-- compare tracking fixtures is more likely to be used at scale


 ** want a parallel debug video generator!!
 ** we'll want to string together end-to-end one day...


class Tracker(object):

  def __init__(self, **hparams):
    ...
  
  def update_and_get_tracks(self, bboxes):
    return bboxes (with track ids and maybe Track objects)
  
  def run_and_profile(self, img_bboxes_df, debugs_output_path=''):

    output new df with:
     * for each image include tracker latency
     * for each bbox include a track_id

    return img_bboxes_df

"""



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
    
  #   mlflow.log_metric('mean_latency_ms', 1e3 * df['detector_latency_sec'].mean())

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
