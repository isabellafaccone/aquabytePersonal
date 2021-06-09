import time

import attr

import numpy as np


###############################################################################
## Postprocessor Impls


DEFAULT_FISH_PARTS = ('UPPER_LIP', 'TAIL_NOTCH', 'DORSAL_FIN', 'PELVIC_FIN')


def asssociate_fish_parts(
        bboxes,
        fish_part_names=DEFAULT_FISH_PARTS,
        fish_category_name='FISH',
        min_assoc_intersect_frac=0.8):
  
  def intersect_score(fish, part):
    return (
      float(fish.get_intersection_with(part).get_area()) / 
        part.get_area())

  fishes = [b for b in bboxes if b.category_name == fish_category_name]
  fish_parts = [b for b in bboxes if b.category_name in fish_part_names]
  completed_fishes = [[fish] for fish in fishes]
  for fish_part in fish_parts:
    best_cluster_idx = -1
    best_cluster_score = -1
    for i, cluster in enumerate(completed_fishes):
      fish = cluster[0]
      cur_score = intersect_score(fish, fish_part)
      if cur_score > best_cluster_score:
        best_cluster_idx = i
        best_cluster_score = cur_score
    
    assert min_assoc_intersect_frac > -1
    if best_cluster_score >= min_assoc_intersect_frac:
      completed_fishes[best_cluster_idx].append(fish_part)
  
  return completed_fishes


def salmon_orthogonality_score(
        fish_bbox,
        part_bboxes,
        expected_parts=DEFAULT_FISH_PARTS):

  if not part_bboxes:
    return 0.
  
  fish_center = np.array(fish_bbox.get_center_xy())
  part_centers = np.array([p.get_center_xy() for p in part_bboxes])
  fish_size = np.array([fish_bbox.width, fish_bbox.height])
  part_dists_scaled = (part_centers - fish_center) / fish_size

  part_distance_score = np.mean(2. * np.abs(part_dists_scaled))
  part_distance_score = min(1., part_distance_score) + 1e-10

  num_expected = len(expected_parts)
  num_parts_score = (
    1. - abs(float(len(part_bboxes)) - num_expected) / num_expected)
  num_parts_score = min(max(num_parts_score, 0.), 1.) + 1e-10

  return 2. / ( 1. / part_distance_score + 1. / num_parts_score)

  



###############################################################################
## Preprocessor Interface and Hooks

class PostrocessorRunner(object):
  def __init__(self):
    self._postprocessors = []

  @classmethod
  def build_from_configs(cls, postprocessor_configs):
    runner = cls()
    for p_conf in postprocessor_configs:
      if p_conf == 'SAOScorer':
        runner._postprocessors.append(SAOScorer())
      else:
        raise ValueError(
          "Don't know how to create postproc %s of %s" % (
            p_conf, postprocessor_configs))
    return runner

  def postprocess(self, img_det):
    pp_to_res_stats = {}
    for p in self._postprocessors:
      start = time.time()
      result = p(img_det)
      p_time = time.time() - start
      pp_to_res_stats[p.get_name()] = (result, p_time)
    return pp_to_res_stats


class PostprocessorBase(object):
  def __call__(self, img_det):
    """Edit img_det in-place and/or return a pickle-able result object
    (e.g. for debugging)"""
    return object()
  
  @classmethod
  def get_name(cls):
    return str(cls.__name__)


@attr.s(slots=True, eq=True)
class SAOScorerResults(object):
  
  fish_clusters = attr.ib(default=attr.Factory(list))
  """bboxes: A list-of-list of `BBox2D` instances, where each sub-list is a
  cluster of bounding boxes for a single fish.  For each inner-list cluster,
  the first bbox is the bbox around the whole fish, and the rest are 
  part associations."""

  # config_str = attr.ib(default="")

  def get_debug_image(self, base_img_src=None):
    import copy
    import cv2
    from mft_utils.bbox2d import BBox2D

    if hasattr(base_img_src, 'load_preprocessed_img'):
      debug_image, _ = base_img_src.load_preprocessed_img()
    elif hasattr(base_img_src, 'shape'):
      debug_image = base_img_src
    else:
      debug_image = None

    for fish_cluster in self.fish_clusters:
      fish = fish_cluster[0]
      parts = fish_cluster[1:]
      
      fish = copy.deepcopy(fish)
      fish.category_name = (
        fish.category_name + ', SAO Score: %s' % fish.extra['SAO_score'])
      debug_image = BBox2D.draw_all_in_img_src(fish_cluster, debug_image)

      # Draw association lines
      fish_center = fish.get_center_xy()
      fish_x, fish_y = int(fish_center[0]), int(fish_center[1])
      for part in parts:
        part_center = part.get_center_xy()
        p_x, p_y = int(part_center[0]), int(part_center[1])
        WHITE_BGR = (255, 255, 255)
        THICKNESS = 3
        cv2.line(
          debug_image, (fish_x, fish_y), (p_x, p_y), WHITE_BGR, THICKNESS)
    
    return debug_image

  def to_html(self, debug_img_src=None):
    import pandas as pd
    from oarphpy.plotting import img_to_data_uri
    
    debug_img = self.get_debug_image(base_img_src=debug_img_src)
    if debug_img is None:
      debug_img_html = "<i>(No fish clusters and no debug image)</i>"
    else:
      w = debug_img.shape[1]
      debug_img_html = """
        <img width="{width}" src="{src}" /><br/>
        <i>To view full resolution: right click and open image in new tab</i>
      """.format(
            src=img_to_data_uri(debug_img),
            width="80%" if w > 800 else w)

    fish_htmls = []
    for fish_cluster in self.fish_clusters:
      fish = fish_cluster[0]

      props = dict(
        (k, getattr(fish, k, None))
        for k in (
          'x', 'y', 'width', 'height',
          'track_id',
          'score',
        )
      )
      for k, v in sorted(fish.extra.items()):
        props['extra.' + k] = str(v)
    
      fish_htmls.append(
        pd.DataFrame([props]).T.style.render())
    
    desc_html = "<br/>".join(fish_htmls)

    return "%s<br/>%s<br/>" % (debug_img_html, desc_html)


class SAOScorer(PostprocessorBase):
  """SAO Score: SAlmon Orthogonality Score"""
  
  def __call__(self, img_det):

    for bbox in img_det.bboxes:
      bbox.extra['SAO_score'] = str(float('nan'))

    completed_fishes = asssociate_fish_parts(img_det.bboxes)
    for completed_fish in completed_fishes:
      fish_bbox, part_bboxes = completed_fish[0], completed_fish[1:]
      score = salmon_orthogonality_score(fish_bbox, part_bboxes)
      fish_bbox.extra['SAO_score'] = str(score)

    return SAOScorerResults(fish_clusters=completed_fishes)


def decode_postproc_result(data):
    # This module contains necessary imports implicitly
    import pickle
    return pickle.loads(data)
