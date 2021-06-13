import typing

import attr

@attr.s(slots=True, eq=True)
class ImgWithBoxes(object):

  ## Core

  img_path = attr.ib(default="")
  """img_path: Path to the image"""

  img_width = attr.ib(default=0)
  """img_width: Width of image in pixels (if known)"""
  
  img_height = attr.ib(default=0)
  """img_height: Height of image in pixels (if known)"""

  bboxes = attr.ib(default=attr.Factory(list))
  """bboxes: A list of `BBox2D` instances"""

  microstamp = attr.ib(default=0)
  """microstamp: Timestamp for this instance in microseconds"""

  ## Stats and Misc (optional)

  detector_latency_sec = attr.ib(type=float, default=-1.)
  """float, optional: Detector latency, if applicable"""

  tracker_latency_sec = attr.ib(type=float, default=-1.)
  """float, optional: Tracker latency, if applicable"""

  bboxes_alt = attr.ib(default=attr.Factory(list))
  """bboxes_alt: A list of *alternative* `BBox2D` instances (e.g. 
      ground truth boxes if this instance has detection boxes)"""

  preprocessor_configs = attr.ib(default=attr.Factory(list))
  """preprocessor_configs: A list of `str` Image preprocessor configurations;
  see below load_preprocessed_img()"""

  postprocessor_configs = attr.ib(default=attr.Factory(list))
  """postprocessors_configs: A list of `str` ImgWithBoxes postprocessor
  configurations; see below run_postprocessors()"""

  postprocessor_to_result = attr.ib(default=attr.Factory(dict))
  """Dict[str, str]: A map of postprocessor -> pickled (result, stats) tuple;
  see run_postprocessors() and get_postprocessor_result() below."""

  extra = attr.ib(default=attr.Factory(dict), type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  @classmethod
  def from_dict(cls, d):
    kwargs = dict(
      (f.name, d[f.name]) for f in attr.fields(cls)
      if f.name in d)
    return cls(**kwargs)

  def get_debug_image(
        self,
        identify_by='category_name',
        alt_identify_by='',
        only_track_id='',
        show_alt=True):
    
    import numpy as np
    if not self.img_path:
      return np.zeros((10, 10, 3))

    import imageio
    # debug = imageio.imread(self.img_path)
    debug, _ = self.load_preprocessed_img()
    for bbox in self.bboxes:
      if only_track_id and bbox.track_id != only_track_id:
        continue
      bbox.draw_in_image(debug, identify_by=identify_by)
    if show_alt:
      for bbox in self.bboxes_alt:
        bbox.draw_in_image(
          debug, identify_by=(alt_identify_by or identify_by), flip_color=True)
    return debug

  def load_preprocessed_img(self):
    import time
    import imageio
    start = time.time()
    img = imageio.imread(self.img_path)
    load_time = time.time() - start
    
    from mft_utils.img_processing import PreprocessorRunner
    p = PreprocessorRunner.build_from_configs(self.preprocessor_configs)
    img, pp_to_stats = p.preprocess(img)
    pp_to_stats['imageio_load'] = load_time
    return img, pp_to_stats

  def run_postprocessors(self):
    from mft_utils.img_bbox_postprocessing import PostrocessorRunner
    p = PostrocessorRunner.build_from_configs(self.postprocessor_configs)
    pp_to_res_stats = p.postprocess(self)

    import pickle
    self.postprocessor_to_result = dict(
      (k, 
        (pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL),
         stats))
      for k, (res, stats) in pp_to_res_stats.items())

  def get_postprocessor_result(self, postproc_name):
    entry = self.postprocessor_to_result.get(postproc_name)
    if entry is None:
      return None
    else:
      v, stats = entry
      from mft_utils.img_bbox_postprocessing import decode_postproc_result
      return decode_postproc_result(v)

  def to_html(self):
    import numpy as np
    debug_img = self.get_debug_image()
    # debug_img_size = max(debug_img.shape[:2])
    # if debug_img_size > 1000:
    #   h, w = debug_img.shape[:2]
    #   if h > w:
    #     scale = 1000. / h
    #   else:
    #     scale = 1000. / w
    #   target_h, target_w = int(scale * h), int(scale * w)

    #   import cv2
    #   debug_img = cv2.resize(debug_img, (target_w, target_h))
    #   print('resized', debug_img.shape)

    from oarphpy.plotting import img_to_data_uri
    w = debug_img.shape[1]
    debug_img_html = """
      <img width="{width}" src="{src}" /><br/>
      <i>To view full resolution: right click and open image in new tab</i>
    """.format(
          src=img_to_data_uri(debug_img),
          width="80%" if w > 800 else w)

    props = {
      'microstamp': self.microstamp,
      'num_boxes': len(self.bboxes),
      'img_path': self.img_path,
      'detector_latency_sec': self.detector_latency_sec,
      'tracker_latency_sec': self.tracker_latency_sec,
      'mean_bbox_score': np.mean([bb.score for bb in self.bboxes] or [-1.]),
    }
    for k, v in sorted(self.extra.items()):
      props['extra.' + k] = str(v)
    
    import pandas as pd
    props_html = pd.DataFrame([props]).T.style.render()

    return "%s<br/>%s<br/>" % (debug_img_html, props_html)
