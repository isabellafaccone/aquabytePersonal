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

  latency_sec = attr.ib(type=float, default=-1.)
  """float, optional: Detector latency, if applicable"""

  bboxes_alt = attr.ib(default=attr.Factory(list))
  """bboxes_alt: A list of *alternative* `BBox2D` instances (e.g. 
      ground truth boxes if this instance has detection boxes)"""

  preprocessor_configs = attr.ib(default=attr.Factory(list))
  """preprocessor_configs: A list of `str` preprocessor configurations; see
  below load_preprocessed_img()"""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  @classmethod
  def from_dict(cls, d):
    kwargs = dict(
      (f.name, d[f.name]) for f in attr.fields(cls)
      if f.name in d)
    return cls(**kwargs)

  def get_debug_image(self):
    import numpy as np
    if not self.img_path:
      return np.zeros((10, 10, 3))

    import imageio
    # debug = imageio.imread(self.img_path)
    debug, _ = self.load_preprocessed_img()
    for bbox in self.bboxes:
      bbox.draw_in_image(debug)
    for bbox in self.bboxes_alt:
      bbox.draw_in_image(debug, flip_color=True)
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

  def to_html(self):
    import numpy as np
    debug_img = self.get_debug_image()

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
      'latency_sec': self.latency_sec,
      'mean_bbox_score': np.mean([bb.score for bb in self.bboxes] or [-1.]),
    }
    for k, v in sorted(self.extra.items()):
      props['extra.' + k] = str(v)
    
    import pandas as pd
    props_html = pd.DataFrame([props]).T.style.render()

    return "%s<br/>%s<br/>" % (debug_img_html, props_html)
