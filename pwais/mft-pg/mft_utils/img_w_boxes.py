import typing

import attr

@attr.s(slots=True, eq=True)
class ImgWithBoxes(object):

  ## Core

  img_path = attr.ib(default="")
  """img_path: Path to the image"""

  bboxes = attr.ib(default=[])
  """bboxes: A list of `BBox2D` instances"""

  microstamp = attr.ib(default=0)
  """microstamp: Timestamp for this instance in microseconds"""

  ## Stats and Misc

  latency_sec = attr.ib(type=float, default=-1)
  """float, optional: Detector latency, if applicable"""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""
