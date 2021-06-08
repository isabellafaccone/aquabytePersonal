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
  part_distance_score = min(1., part_distance_score)

  num_parts_score = float(len(part_bboxes)) / len(expected_parts)
  num_parts_score = min(1., num_parts_score)

  return 2. / (part_distance_score + num_parts_score)

  



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
  
  # config_str = attr.ib(default="")

  # TODO: to_html() for plots


class SAOScorer(PostprocessorBase):
  """SAO Score: SAlmon Orthogonality Score"""
  
  def __call__(self, img_det):

    for bbox in img_det.bboxes:
      bbox.extra['SAO_score'] = str('0')

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
