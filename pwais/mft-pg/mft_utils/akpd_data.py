import copy
import os

import pandas as pd

from mft_utils import misc as mft_misc
from mft_utils.bbox2d import BBox2D
from mft_utils.img_w_boxes import ImgWithBoxes

def create_cleaned_df(
    in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
    imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/'):
  """Clean the given AKPD Ground Truth CSV data and return a
  cleaned DataFrame."""

  import os

  import imageio
  import pandas as pd

  df_in = pd.read_csv(in_csv_path)

  # LOL those annotations aren't JSONs, they're string-ified python dicts :(
  import ast

  rows_out = []
  images_seen = set()
  for i, row in enumerate(df_in.to_dict(orient='records')):
    t = pd.to_datetime(row['captured_at'])
    microstamp = int(1e6 * t.timestamp())
    
    keypoints_raw = row['keypoints']
    keypoints_dict = ast.literal_eval(keypoints_raw)

    base_extra = {
      'akpd.camera_metadata': row['camera_metadata'], # is a str
    }

    for camera in ['left', 'right']:
      
      img_s3_uri = row[camera + '_image_url']
      if 'https://s3-eu-west-1.amazonaws.com/aquabyte-crops/' in img_s3_uri:
        img_path = img_s3_uri.replace('https://s3-eu-west-1.amazonaws.com/aquabyte-crops/', imgs_basedir)
      elif 'https://aquabyte-crops.s3.eu-west-1.amazonaws.com/' in img_s3_uri:
        img_path = img_s3_uri.replace('https://aquabyte-crops.s3.eu-west-1.amazonaws.com/', imgs_basedir)
      elif 'http://aquabyte-crops.s3.eu-west-1.amazonaws.com/' in img_s3_uri:
        img_path = img_s3_uri.replace('http://aquabyte-crops.s3.eu-west-1.amazonaws.com/', imgs_basedir)
      elif 's3://aquabyte-crops/' in img_s3_uri:
        img_path = img_s3_uri.replace('s3://aquabyte-crops/', imgs_basedir)
      else:
        raise ValueError(img_s3_uri)

      assert os.path.exists(img_path), img_path

      if img_path in images_seen:
        print(img_path, 'is a dupe')
        continue
      images_seen.add(img_path)

      img = imageio.imread(img_path)
      h, w = img.shape[:2]

      camera_kps = keypoints_dict[camera + 'Crop']
      kp_df = pd.DataFrame(camera_kps)

      crop_meta = ast.literal_eval(row[camera + '_crop_metadata'])
      quality = crop_meta.get('qualityScore', {})
      rows_out.append({
        'camera': camera,
        'captured_at': pd.to_datetime(row['captured_at']),
        'img_path': img_path,
        'img_height': h,
        'img_width': w,
        'keypoints': kp_df,
        'crop_meta_raw': row[camera + '_crop_metadata'], # str
        'camera_metadata': row['camera_metadata'], # str
        'quality': str(quality.get('quality', float('nan'))),
        'blurriness': str(quality.get('blurriness', float('nan'))),
        'darkness': str(quality.get('darkness', float('nan'))),
        'mean_luminance': str(crop_meta.get('mean_luminance', float('nan'))),
      })
    
    if (i+1) % 100 == 0:
      print("... cleaned %s of %s ..." % (i+1, len(df_in)))
  
  return pd.DataFrame(rows_out)


def get_akpd_as_bbox_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/',
      kp_bbox_to_fish_scale=0.1,
      only_camera='left',
      only_parts=[],
      parallel=-1):

  cleaned_df_path = in_csv_path + 'cleaned.pkl'
  if not os.path.exists(cleaned_df_path):
    mft_misc.log.info("Cleaning labels and caching cleaned copy ...")
    df = create_cleaned_df(in_csv_path=in_csv_path, imgs_basedir=imgs_basedir)
    df.to_pickle(cleaned_df_path)
  
  mft_misc.log.info("Using cached / cleaned labels at %s" % cleaned_df_path)
  labels_df = pd.read_pickle(cleaned_df_path)
  if only_camera:
    labels_df = labels_df[labels_df['camera'] == only_camera]
  mft_misc.log.info("Have %s labels" % len(labels_df))

  def to_img_gt(row):
    w, h = row['img_width'], row['img_height']
    bboxes = [
      BBox2D(
        x=kp['xCrop'], width=1, im_width=w,
        y=kp['yCrop'], height=1, im_height=h,
        category_name=kp['keypointType'])
      for kp in row['keypoints'].to_dict(orient='records')
    ]

    # Size the keypoints relative to the size of the image / crop (relative
    # to the size of the fish in pixels)
    padding = int(kp_bbox_to_fish_scale * w), int(kp_bbox_to_fish_scale * h)
    for bb in bboxes:
      bb.add_padding(*padding)
      bb.clamp_to_screen()

    img_path = row['img_path']
    microstamp = int(1e6 * row['captured_at'].timestamp())
    extra = {
      # 'akpd.camera_metadata': row['camera_metadata'], # str
      'camera': row['camera'],
      'akpd.quality': str(row['quality']),
      'akpd.blurriness': str(row['blurriness']),
      'akpd.darkness': str(row['darkness']),
      'akpd.mean_luminance': str(row['mean_luminance']),
    }
    return ImgWithBoxes(
                      img_path=img_path,
                      img_width=w,
                      img_height=h,
                      bboxes=bboxes,
                      microstamp=microstamp,
                      extra=extra)

  if parallel < 0:
    parallel = os.cpu_count()
  img_gts = mft_misc.foreach_threadpool_safe_pmap(
              to_img_gt,
              labels_df.to_dict(orient='records'),
              {'max_workers': parallel})
  return img_gts


# def get_akpd_as_bbox_img_gts_with_ablated_parts(
#       in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
#       imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/',
#       kp_bbox_to_fish_scale=0.1,
#       only_camera='left',
#       only_keypoints=[],
#       parallel=-1):



class SyntheticAKPDFromBBoxesGenerator(object):

  ASPECT_RATIO_EPS = 1e-3
  N_TRIALS = 10
  # MINIMUM_VISIBILITY_FRAC_FOR_VISIBLE = 0.85
 
  def __init__(self, akpd_img_gts):
    self._akpd_img_gts = akpd_img_gts

    aspect_ratio_to_idx = self._get_aspect_ratio_to_idx()
    ordered = sorted(aspect_ratio_to_idx.items())
    self._aspects = [k for k, v in ordered]
    self._img_ids = [v for k, v in ordered]

  def create_synth_img_gt(self, base_image, fish_bboxes):
    """Create and return a synthesized (image, `ImgWithBoxes`) tuple
    from the given `base_image` and target `fish_bboxes`.  We assume that
    `fish_bboxes` are implicitly topologically sorted by Z with the 
    nearest box descending-- the first bbox(es) in `fish_bboxes` should
    be unoccluded, with possibly full-occluded boxes at the very end.
    """

    # Pick AKPD fish to use in the synthetic image
    aspect_ratios = [float(b.width)/ b.height for b in fish_bboxes]
    chosen = self._sample_best_matches(aspect_ratios)

    # Now paste each AKPD fish, starting with the most distance fish
    aug_img = base_image.copy()
    aug_img_gt = ImgWithBoxes()
    for i in reversed(range(len(fish_bboxes))):
      dest_bbox = fish_bboxes[i]
      akpd_img_id = chosen[i]
      self._paste_in_place(
        dest_bbox,
        aug_img,
        aug_img_gt,
        akpd_img_id)
    
    self._winnow_occluded(aug_img_gt)

    return aug_img, aug_img_gt

  def _get_aspect_ratio_to_idx(self):
    aspect_ratio_to_idx = {}
    for idx, img_gt in enumerate(self._akpd_img_gts):
      aspect_ratio = float(img_gt.img_width) / img_gt.img_height
      for trial in range(self.N_TRIALS):
        if aspect_ratio in aspect_ratio_to_idx:
          aspect_ratio += self.ASPECT_RATIO_EPS # Try to include every image
        else:
          break
      aspect_ratio_to_idx[aspect_ratio] = idx
    return aspect_ratio_to_idx

  def _get_idx_of_best_match(self, aspect_ratio):
    import bisect
    nearest = bisect.bisect_left(self._aspects, aspect_ratio)
    
    left = max(0, nearest - 1)
    right = min(nearest, len(self._aspects) - 1)

    left_dist = abs(self._aspects[left] - aspect_ratio)
    right_dist = abs(self._aspects[right] - aspect_ratio)
    if left_dist < right_dist:
      img_id = self._img_ids[left]
    else:
      img_id = self._img_ids[right]
    return img_id
  
  def _sample_best_matches(self, aspect_ratios):
    chosen = []
    for aspect_ratio in aspect_ratios:
      img_id = None
      for trial in range(self.N_TRIALS):
        img_id = self._get_idx_of_best_match(aspect_ratio)
        if img_id in chosen:
          aspect_ratio += self.ASPECT_RATIO_EPS
        else:
          break
      assert img_id is not None
      chosen.append(img_id)
    return chosen

  def _paste_in_place(
        self,
        dest_bbox,
        dest_img,
        aug_img_gt,
        akpd_img_id):
    
    akpd_img_gt = self._akpd_img_gts[akpd_img_id]
    akpd_img, _ = akpd_img_gt.load_preprocessed_img()

    import cv2
    target_fish_width, target_fish_height = dest_bbox.width, dest_bbox.height
    akpd_img = cv2.resize(akpd_img, (target_fish_width, target_fish_height))

    # Paste fish image!
    dx1, dy1, dx2, dy2 = dest_bbox.get_x1_y1_x2_y2()
    dest_img[dy1:dy2+1, dx1:dx2+1, :3] = akpd_img[:, :, :3]

    bboxes_to_add = []

    # Paste a FISH box ...
    fish_bbox = copy.deepcopy(dest_bbox)
    fish_bbox.category_name = 'AKPD_SYNTH_FISH'
    bboxes_to_add.append(fish_bbox)
    
    # ... and paste fish part boxes
    dh, dw = dest_img.shape[:2]
    for bbox in akpd_img_gt.bboxes:
      part_bbox = copy.deepcopy(bbox)
      part_bbox = part_bbox.get_rescaled_to_target_image(
                            target_fish_width, target_fish_height)
      part_bbox.x += dest_bbox.x
      part_bbox.y += dest_bbox.y
      part_bbox.im_width = dw
      part_bbox.im_height = dh
      bboxes_to_add.append(part_bbox)

    for bbox in bboxes_to_add:
      bbox.extra['akpd_synth.src_img_path'] = akpd_img_gt.img_path
      bbox.extra['akpd_synth.img_fish_id'] = str(akpd_img_id)

      for k, v in akpd_img_gt.extra.items():
        bbox.extra['akpd_synth.src_img_extra.' + k] = v

      bbox.extra['_akpd_synth_occluders'] = []

    # Note anything that the new fish occludes
    for existing_bbox in aug_img_gt.bboxes:
      if existing_bbox.category_name == 'AKPD_SYNTH_FISH':
        continue
      if fish_bbox.overlaps_with(existing_bbox):
        existing_bbox.extra['_akpd_synth_occluders'].append(fish_bbox)
      # for bbox in bboxes_to_add:
      #   if bbox.overlaps_with(existing_bbox):
      #     existing_bbox.extra['_akpd_synth_occluders'].append(bbox)

    aug_img_gt.bboxes += bboxes_to_add
  
  def _winnow_occluded(self, aug_img_gt):
    import numpy as np
    
    boxes_to_keep = []
    for bbox in aug_img_gt.bboxes:
      visible = np.ones((bbox.width, bbox.height))
      for occluder in bbox.extra['_akpd_synth_occluders']:
        intersection = bbox.get_intersection_with(occluder)
        xmin, ymin, xmax, ymax = intersection.get_x1_y1_x2_y2()
        xmin -= bbox.x
        xmax -= bbox.x
        ymin -= bbox.y
        ymax -= bbox.y
        visible[xmin:xmax+1, ymin:ymax+1] = 0

      if not visible.any():
        # Skip this box
        continue

      num_visible = int(visible.sum())
      bbox.extra['akpd_synth.num_px_visible'] = str(num_visible)

      bbox.extra['akpd_synth.is_occluded'] = str(
                          bool(bbox.extra['_akpd_synth_occluders']))
      del bbox.extra['_akpd_synth_occluders']
      
      boxes_to_keep.append(bbox)

    aug_img_gt.bboxes = boxes_to_keep

    aug_img_gt.extra['akpd_synth.num_fish_with_occluded'] = str(
                            sum(
                              1
                              for bb in aug_img_gt.bboxes
                              if bb.extra['akpd_synth.is_occluded'] == 'True'))


        

    # if nearest_idx == len(self._aspects):
    #   nearest_idx -= 1
    
    # right_dist = abs(self._aspects[nearest_idx + 1] - aspect_ratio)
    # left_dist = abs(self._aspects[nearest_idx] - aspect_ratio)
    
    # if right_dist < left_dist:
    #   nearest_idx -= 1

    # if 0 < nearest <= len()


#   """
#      * cache a map of aspect ratio -> image path.  let OS disk cache do the actual image caching
#      * accept a topo-sorted list of bboxes.  bbox at front should always be z-ordered
#          closer than bbox at end
     
#      -- every time paste a fish, then need to filter / occlude / pop some bbox labels. 
#        PROBABLY just pop the fish parts.  maybe also BLUR those pixels!


# https://github.com/albumentations-team/albumentations/blob/1a35d2c4198b63dbcffa3ecff15d17b0f66d7fd2/albumentations/augmentations/transforms.py#L1055

#   """
#   pass

def generate_synth_fish_and_parts(
        output_dir='/opt/mft-pg/datasets/datasets_s3/synth_fish_and_parts',
        kp_bbox_to_fish_scale=0.1,
        synth_img_width=4096,
        synth_img_height=3000,
        only_parts=None):
  
  if not only_parts:
    only_parts = ('UPPER_LIP', 'TAIL_NOTCH', 'DORSAL_FIN', 'PELVIC_FIN')

  akpd_img_gts = get_akpd_as_bbox_img_gts(
                    kp_bbox_to_fish_scale=kp_bbox_to_fish_scale)
  for akpd_img_gt in akpd_img_gts:
    akpd_img_gt.bboxes = [
      bb for bb
      in akpd_img_gt.bboxes
      if bb.category_name in only_parts
    ]
  gen = SyntheticAKPDFromBBoxesGenerator(akpd_img_gts)

  from mft_utils import high_recall_fish_data as hrf_data
  himg_gts = hrf_data.get_img_gts()

  class Renderer(object):
    def __init__(self, gen, himg_gts):
      self._gen = gen
      self._himg_gts = himg_gts

    def get_image_ids(self):
      return list(range(len(self._himg_gts)))
    
    def render_synth_img(self, img_id):
      import os 
      from mft_utils import misc as mft_misc
      
      himg_gt = self._himg_gts[img_id]

      base_image, _ = himg_gt.load_preprocessed_img()

      # Convert back to target resolution (e.g. native / AKPD)
      import cv2
      base_image = cv2.resize(base_image, (synth_img_width, synth_img_height))
      
      # Order ground truth by Z-order
      full_fish = []
      partial_fish = []
      for bbox in himg_gt.bboxes:
        rbbox = bbox.get_rescaled_to_target_image(
                                  synth_img_width, synth_img_height)
        if rbbox.extra['is_partial'] == 'True':
          partial_fish.append(rbbox)
        else:
          full_fish.append(rbbox)
      fish_bboxes = full_fish + partial_fish
      aug_img, aug_img_gt = self._gen.create_synth_img_gt(
                                          base_image, fish_bboxes)
      
      mft_misc.mkdir(output_dir)
      dest_img_path = os.path.join(output_dir, 'example_%s.png' % img_id)
      dest_labels_path = dest_img_path + '.aug_img_gt.pkl'

      import imageio
      imageio.imwrite(dest_img_path, aug_img)

      aug_img_gt.extra['akpd_synth.base_src_img_path'] = himg_gt.img_path
      aug_img_gt.img_path = dest_img_path
      aug_img_gt.extra.update(himg_gt.extra)
        
      with open(dest_labels_path, 'wb') as f:
        import pickle
        pickle.dump(aug_img_gt, f, protocol=pickle.HIGHEST_PROTOCOL)

      mft_misc.log.info("Saved synth %s" % dest_labels_path)
  
  r = Renderer(gen, himg_gts)
  img_ids = r.get_image_ids()

  from oarphpy import spark as S
  class MFTSpark(S.SessionFactory):
    pass
  
  mft_misc.log.info("Going to create %s synthetic images ..." % len(img_ids))
  with MFTSpark.sess() as spark:
    task_rdd = spark.sparkContext.parallelize(img_ids, numSlices=len(img_ids))
    task_rdd.foreach(lambda img_id: r.render_synth_img(img_id))

  
def get_synth_fish_and_parts_img_gts(
        data_dir='/opt/mft-pg/datasets/datasets_s3/synth_fish_and_parts',
        parallel=-1):
  
  paths = [
    os.path.join(data_dir, fname)
    for fname in os.listdir(data_dir)
    if fname.endswith('.aug_img_gt.pkl')
  ]

  def load_img_gt(path):
    import pickle
    
    from mft_utils.bbox2d import BBox2D
    from mft_utils.img_w_boxes import ImgWithBoxes

    with open(path, 'rb') as f:
      return pickle.load(f)

  if parallel < 0:
    parallel = os.cpu_count()
  img_gts = mft_misc.foreach_threadpool_safe_pmap(
              load_img_gt,
              paths,
              {'max_workers': parallel})
  mft_misc.log.info(
    "Loaded %s records for synth_fish_and_parts" % len(img_gts))
  return img_gts


DATASET_NAME_TO_ITER_FACTORY = {

  ## NB: At the time of writing, the first ~4420 examples have quality / darkness
  ## scores, so we save those examples for the test set.
  ## TODO do a fresh shuffle 
  'akpd1.0_scale0.1_left_test': (lambda:
      get_akpd_as_bbox_img_gts(
        kp_bbox_to_fish_scale=0.1,
        only_camera='left')[:4200]),
  'akpd1.0_scale0.1_left_train': (lambda:
      get_akpd_as_bbox_img_gts(
        kp_bbox_to_fish_scale=0.1,
        only_camera='left')[4200:]),
  
  'akpd1.0_scale0.05_left_test': (lambda:
      get_akpd_as_bbox_img_gts(
        kp_bbox_to_fish_scale=0.05,
        only_camera='left')[:4200]),
  'akpd1.0_scale0.05_left_train': (lambda:
      get_akpd_as_bbox_img_gts(
        kp_bbox_to_fish_scale=0.05,
        only_camera='left')[4200:]),
    
  'akpd1.0_synth_fish_and_parts.0.1.train': (lambda :
      get_synth_fish_and_parts_img_gts()[:4980]),
  'akpd1.0_synth_fish_and_parts.0.1.test': (lambda :
      get_synth_fish_and_parts_img_gts()[4980:]),
}

if __name__ == '__main__':
  generate_synth_fish_and_parts()
