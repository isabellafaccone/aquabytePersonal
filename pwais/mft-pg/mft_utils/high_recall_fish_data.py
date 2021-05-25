def create_cleaned_df(
    in_csv_path='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv',
    imgs_basedir='/opt/mft-pg/datasets/datasets_s3/high_recall_fish1/images/'):
  """Clean the given AKPD Ground Truth CSV data and return a
  cleaned DataFrame."""

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
  n_partial = 0
  n_annotator_skipped = 0
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

    assert os.path.exists(img_path), img_path

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
      matches = re.findall(r'pen_id=(.*)/', img_s3_uri)
      pen_id = matches[0]
      timestamp = pd.to_datetime(matches[0])
    except Exception:
      pen_id = 'unknown'  
    
    annos_raw = ast.literal_eval(row['annotation'])
    annotation_is_partial = bool(annos_raw.get('isPartial'))
    n_partial += 1

    skip_reasons = annos_raw.get('skipReasons', [])
    if 'annotations' not in annos_raw:
      n_annotator_skipped += 1
      bboxes = None
    else:
      bboxes = pd.DataFrame(annos_raw['annotations'])

    metadata_raw = ast.literal_eval(row['metadata'])
    meta_tags = metadata_raw.get('tags', [])

    rows_out.append({
      'img_path': img_path,
      'img_height': h,
      'img_width': w,
      'pen_id': pen_id,
      'timestamp': timestamp,
      'annotation_is_partial': annotation_is_partial,
      'bboxes': bboxes,
      'meta_tags': meta_tags,
    })
    
    if (i+1) % 100 == 0:
      print("... cleaned %s of %s ..." % (i+1, len(df_in)))
  
  print('n_duplicates', n_duplicates)
  print('n_partial', n_partial)
  print('n_annotator_skipped', n_annotator_skipped)
  print('len(rows_out)', len(rows_out))
  return pd.DataFrame(rows_out)


