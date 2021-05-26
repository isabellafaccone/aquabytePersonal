
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
}
