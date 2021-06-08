import attr

import numpy as np


###############################################################################
## Postprocessor Impls

@attr.s(slots=True, eq=True)
class FishPartAssociations(object):
  
  bbox_clusters = attr.ib(default=attr.Factory(list))
  
  assoc_config_str = attr.ib(default="")

  score_config_str = attr.ib(default="")


DEFAULT_FISH_PARTS = ('UPPER_LIP', 'TAIL_NOTCH', 'DORSAL_FIN', 'PELVIC_FIN')

def asssociate_fish_parts(
        bboxes,
        fish_part_names=DEFAULT_FISH_PARTS,
        fish_category_name='FISH',
        min_IOU_assoc_part_to_fish=0.8):
  
  def IOU(a, b):
    return (
      float(a.get_intersection_with(b).get_area()) / 
        a.get_union_with(b).get_area())

  fishes = [b for b in bboxes if b.category_name == fish_category_name]
  fish_parts = [b for b in bboxes if b.category_name in fish_part_names]
  completed_fish = [[fish] for fish in fishes]
  for fish_part in fish_parts:
    best_cluster = -1
    best_IOU = -1
    for i, cluster in enumerate(completed_fish):
      fish = cluster[0]
      cur_iou = IOU(fish, fish_part)
      if cur_iou > best_IOU:
        best_cluster = i
        best_IOU = cur_iou
    
    assert min_IOU_assoc_part_to_fish > -1
    if best_IOU >= min_IOU_assoc_part_to_fish:
      completed_fish[best_cluster].append(fish_part)
  
  return completed_fish


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

  part_distance_score = np.mean(2. * part_dists_scaled.abs())
  part_distance_score = max(1., part_distance_score)

  num_parts_score = float(len(part_bboxes)) / len(expected_parts)
  num_parts_score = max(1., num_parts_score)

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
      if p_conf == 'FishPoseScorer':
        runner._preprocessors.append(FishPoseScorer())
      else:
        raise ValueError(
          "Don't know how to create preproc %s of %s" % (
            p_conf, postprocessor_configs))
    return runner

  def postprocess(self, img_det):
    import time
    pp_to_res_stats = {}
    for p in self._preprocessors:  
      start = time.time()
      result = p(img_det)
      p_time = time.time() - start
      pp_to_res_stats[p.get_name()] = (result, p_time)
    return pp_to_res_stats


class PostprocessorBase(object):
  def __call__(self, img_det):
    return object()
  
  @classmethod
  def get_name(cls):
    return str(cls.__name__)

class SAOScorer(PostprocessorBase):
  def __call__(self, img_det):
    asssociate_fish_parts(img_det)
    asssociate_fish_parts(img_det)
    return object()
