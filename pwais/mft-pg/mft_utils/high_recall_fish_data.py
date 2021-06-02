import os

import pandas as pd

from mft_utils import misc as mft_misc
from mft_utils.bbox2d import BBox2D
from mft_utils.img_w_boxes import ImgWithBoxes



def create_cleaned_df(
    in_csv_path='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv',
    imgs_basedir='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/images/'):
  """Clean the given Ground Truth CSV data and return a cleaned DataFrame."""

  import os
  import re

  import imageio
  import pandas as pd

  df_in = pd.read_csv(in_csv_path)

  # LOL those annotations aren't JSONs, they're string-ified python dicts :(
  import ast

  rows_out = []
  images_seen = set()
  n_duplicates = 0
  n_annotator_skipped = 0
  n_missing_images = 0
  for i, row in enumerate(df_in.to_dict(orient='records')):
    
    img_lst = ast.literal_eval(row['images'])
    if not img_lst:
      print('bad anno', row)
      continue
      
    img_s3_uri = img_lst[0]
    if 's3://aquabyte-frames-resized-inbound/' in img_s3_uri:
      img_path = img_s3_uri.replace('s3://aquabyte-frames-resized-inbound/', imgs_basedir)
    else:
      raise ValueError(img_s3_uri)

    if not os.path.exists(img_path):
      print('missing image or bad uri', img_path, img_s3_uri)
      n_missing_images += 1
      continue

    if img_path in images_seen:
      print(img_path, 'is a dupe')
      n_duplicates += 1
      continue
    images_seen.add(img_path)

    img = imageio.imread(img_path)
    h, w = img.shape[:2]
    
    # try to deduce timestamp, the url usually has a substring in the path like:
    # pen-id=95/date=2020-06-24/hour=12/at=2020-06-24T12:01:30.925097000Z/
    try:
      matches = re.findall(r'at=(.*)/', img_s3_uri)
      timestamp = pd.to_datetime(matches[0])
    except Exception:
      timestamp = pd.to_datetime(0)
    
    try:
      matches = re.findall(r'/pen[_-]id=(\d+)/', img_s3_uri)
      pen_id = int(matches[0])
    except Exception:
      pen_id = 'unknown'
    
    annos_raw = ast.literal_eval(row['annotation'])
    # NB: the isPartial thing is junk from a previous version of the labeler
    # annotation_is_partial = bool(annos_raw.get('isPartial'))
    # if annotation_is_partial:
    #   n_partial += 1

    skip_reasons = annos_raw.get('skipReasons', [])
    
    if 'annotations' not in annos_raw:
      n_annotator_skipped += 1
      bboxes = None
    else:
      bboxes = pd.DataFrame(annos_raw['annotations'])
        # >>> df['bboxes']
        # 0             label  width  xCrop  yCrop  height catego...
        #                                ...                        
        # 10929                                                 None
        # 10930         label  width  xCrop  yCrop  height catego...
        # >>> df['bboxes'][10930]
        #      label  width  xCrop  yCrop  height category
        # 0     FULL    214    181    264     120     FISH
        # 1     FULL    128    379    205      54     FISH
        # 2  PARTIAL     78     79    159      46     FISH
        # 3     FULL     87     76    189      41     FISH
        # 4     FULL     91    305     50      31     FISH
        # 5     FULL     53    289    187      33     FISH

    metadata_raw = ast.literal_eval(row['metadata'])
    meta_tags = metadata_raw.get('tags', [])

    if 'left_frame' in img_path:
      camera = 'left'
    elif 'right_frame' in img_path:
      camera = 'right'
    else:
      camera = 'unknown'

    rows_out.append({
      'camera': camera,
      'img_path': img_path,
      'img_height': h,
      'img_width': w,
      'pen_id': pen_id,
      'timestamp': timestamp,
      'bboxes': bboxes,
      'meta_tags': meta_tags,
      'skip_reasons': skip_reasons,
    })
    
    if (i+1) % 100 == 0:
      print("... cleaned %s of %s ..." % (i+1, len(df_in)))
  
  print('n_duplicates', n_duplicates)
  print('n_annotator_skipped', n_annotator_skipped)
  print('n_missing_images', n_missing_images)
  print('len(rows_out)', len(rows_out))
  return pd.DataFrame(rows_out)


def get_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/images/',
      only_camera='left',
      parallel=-1):

  # NB original image dimensions are 4096x3000

  cleaned_df_path = in_csv_path + '.cleaned.pkl'
  if not os.path.exists(cleaned_df_path):
    mft_misc.log.info("Cleaning labels and caching cleaned copy ...")
    df = create_cleaned_df(in_csv_path=in_csv_path, imgs_basedir=imgs_basedir)
    df.to_pickle(cleaned_df_path)
  
  mft_misc.log.info("Using cached / cleaned labels at %s" % cleaned_df_path)
  labels_df = pd.read_pickle(cleaned_df_path)
  if only_camera:
    labels_df = labels_df[labels_df['camera'] == only_camera]
  mft_misc.log.info("Have %s labeled images" % len(labels_df))

  def to_img_gt(row):
    w, h = row['img_width'], row['img_height']
    if row['bboxes'] is None:
      bboxes = []
    else:
      bboxes = [
        BBox2D(
          x=bb['xCrop'], width=bb['width'], im_width=w,
          y=bb['yCrop'], height=bb['height'], im_height=h,
          category_name=bb['category'],
          extra={'is_partial': bb['label'] == 'PARTIAL'})
        for bb in row['bboxes'].dropna().to_dict(orient='records')
      ]

    bboxes = [
      bb for bb in bboxes
      if (bb.width >= 1 and bb.height >= 1)
    ]

    for bb in bboxes:
      bb.clamp_to_screen()

    img_path = row['img_path']
    microstamp = int(1e6 * row['timestamp'].timestamp())
    extra = {
      'camera': row['camera'],
      'pen_id': row['pen_id'],
      'labeler.skip_reasons': str(row['skip_reasons']),
      'labeler.meta_tags': str(row['meta_tags']),
    }
    return ImgWithBoxes(
                      img_path=img_path,
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

def get_img_gts_clahe(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/images/',
      only_camera='left',
      parallel=-1):
  
  cache_base = '/tmp/mft_pg_high_recall_fish1_clahe'
  if not os.path.exists(cache_base):
    mft_misc.log.info("Generating CLAHE cache to ... %s " % cache_base)

    src = imgs_basedir
    if not src.endswith('/'):
      src = src + '/'
    dest = cache_base
    if not dest.endswith('/'):
      dest = dest + '/'
    mft_misc.run_cmd("rsync -a -v -h --size-only --progress %s %s" % (src, dest))

    from oarphpy.util import all_files_recursive
    paths = all_files_recursive(cache_base, pattern="*.jpg")
    
    def clahe_in_place(path):
      import imageio
      from mft_utils import img_processing as improc
      img = imageio.imread(path)
      img = improc.CLAHE_enhance(img)
      imageio.imwrite(path, img)
    
    if parallel < 0:
      parallel = os.cpu_count()

    mft_misc.log.info("Applying to %s paths ... " % len(paths))
    mft_misc.foreach_threadpool_safe_pmap(
              clahe_in_place,
              paths,
              {'max_workers': parallel})
    mft_misc.log.info("... done with CLAHE!")
  
  return get_img_gts(
            in_csv_path=in_csv_path,
            imgs_basedir=cache_base,
            only_camera=only_camera,
            parallel=parallel)


def with_clahe(img_gt):
  img_gt.preprocessor_configs = ['clahe']
  return img_gt


DATASET_NAME_TO_ITER_FACTORY = {

  ## NB: At the time of writing, the first ~4420 examples have quality / darkness
  ## scores, so we save those examples for the test set.
  ## TODO do a fresh shuffle 
  'hrf_1.0_train': (lambda: get_img_gts()[:5400]),
  'hrf_1.0_test': (lambda: get_img_gts()[5400:]),

  'hrf_clahe_1.0_train': (lambda: get_img_gts_clahe()[:5400]),
  'hrf_clahe_1.0_test': (lambda: [
    with_clahe(img_gt) for img_gt in get_img_gts()[5400:]
      # Apply CLAHE at inference time so that we can time it on the TX2
  ]),

}
