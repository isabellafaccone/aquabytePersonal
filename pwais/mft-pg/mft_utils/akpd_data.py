
def get_akpd_as_bbox_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/',
      kp_bbox_to_fish_scale=0.1,
      only_camera='left'):
  
  import csv
  import os
  from datetime import datetime

  # LOL those annotations aren't JSONs, they're string-ified python dicts :(
  import ast

  from mft_utils.bbox2d import BBox2D
  from mft_utils.img_w_boxes import ImgWithBoxes

  with open(in_csv_path, newline='') as f:
    rows = list(csv.DictReader(f))
  
  if not imgs_basedir.endswith('/'):
    imgs_basedir = imgs_basedir + '/'

  img_gts_out = []
  for row in rows:

    timestamp_raw = row['captured_at']
    if '+' in timestamp_raw:
      timestamp_raw = timestamp_raw.split('+')[0]
    try:
      utc_time = datetime.strptime(timestamp_raw, "%Y-%m-%d %H:%M:%S.%f")
    except:
      utc_time = datetime.strptime(timestamp_raw, "%Y-%m-%d %H:%M:%S")
      
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
    microstamp = int(epoch_time * 1e6)

    keypoints_raw = row['keypoints']
    keypoints_dict = ast.literal_eval(keypoints_raw)

    base_extra = {
      'akpd.camera_metadata': row['camera_metadata'], # is a str
    }

    if only_camera:
      cameras = [only_camera]
    else:
      cameras = ['left', 'right']
    
    for camera in cameras:
      
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

      crop_meta = ast.literal_eval(row[camera + '_crop_metadata'])
      camera_kps = keypoints_dict[camera + 'Crop']
      w, h = crop_meta['width'], crop_meta['height']
      bboxes = [
        BBox2D(
          x=kp['xCrop'], width=1, im_width=w,
          y=kp['yCrop'], height=1, im_height=h,
          category_name=kp['keypointType'])
        for kp in camera_kps
      ]

      # Size the keypoints relative to the size of the image / crop (relative
      # to the size of the fish in pixels)
      padding = int(kp_bbox_to_fish_scale * w), int(kp_bbox_to_fish_scale * h)
      for bb in bboxes:
        bb.add_padding(*padding)
        bb.clamp_to_screen()

      quality = crop_meta.get('qualityScore', {})
      extra = dict(base_extra)
      extra.update({
        'akpd.crop_metadata_raw': row[camera + '_crop_metadata'], # is a str
        'akpd.quality': str(quality.get('quality', float('nan'))),
        'akpd.blurriness': str(quality.get('blurriness', float('nan'))),
        'akpd.darkness': str(quality.get('darkness', float('nan'))),
        'akpd.mean_luminance': str(crop_meta.get('mean_luminance', float('nan'))),
      })

      img_gts_out.append(
                    ImgWithBoxes(
                      img_path=img_path,
                      bboxes=bboxes,
                      microstamp=microstamp))
  return img_gts_out

DATASET_NAME_TO_FACTORY = {
  'akpd1.0_scale0.1_left_train': (lambda:
    get_akpd_as_bbox_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/',
      kp_bbox_to_fish_scale=0.1,
      only_camera='left')[:4885]),
  'akpd1.0_scale0.1_left_test': (lambda:
    get_akpd_as_bbox_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/akpd1/images/',
      kp_bbox_to_fish_scale=0.1,
      only_camera='left')[4885:]),
}
