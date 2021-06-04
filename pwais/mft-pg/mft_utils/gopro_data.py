
# Determined manually from GoPro video source.  Probably a file named
# `gopro_fish_footage.mp4`
GOPRO1_FPS = 60.

def iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=[]):

  # NB: adapted from convert_to_yolo_format() in mft-pg
  # Some day mebbe refactor to join them...

  import os
  import csv

  from mft_utils.bbox2d import BBox2D
  from mft_utils.img_w_boxes import ImgWithBoxes
  from mft_utils import misc as mft_misc
  

  with open(in_csv_path, newline='') as f:
    rows = list(csv.DictReader(f))

  mft_misc.log.info('Read %s rows from %s' % (len(rows), in_csv_path))

  # Read and cache the image dims only once; they're from a video so all
  # the same
  h = None
  w = None

  T_STEP_MICROS = int((1. / GOPRO1_FPS) * 1e6)
  for row in rows:

    img_fname = os.path.basename(row['image_f'])
    img_path = os.path.join(imgs_basedir, img_fname)

    if (h, w) == (None, None):
      import imageio

      img = imageio.imread(img_path)
      h, w = img.shape[:2]
      
      mft_misc.log.info("Images have dimensions width %s height %s" % (w, h))

    # LOL those annotations aren't JSONs, they're string-ified python dicts :(
    import ast
    annos_raw = ast.literal_eval(row['annotation'])
    bboxes = []
    for anno in annos_raw['annotations']:
      if only_classes and anno['category'] not in only_classes:
        continue

      if 'xCrop' not in anno:
        print('bad anno %s' % (row,))
        continue

      x_pixels = anno['xCrop']
      y_pixels = anno['yCrop']
      w_pixels = anno['width']
      h_pixels = anno['height']

      bboxes.append(
        BBox2D(
          category_name=anno['category'],
          x=x_pixels,
          y=y_pixels,
          width=w_pixels,
          height=h_pixels,
          im_width=w,
          im_height=h))

    f_num = int(img_fname.lstrip('frame_').rstrip('.jpg'))
    microstamp = int(f_num * T_STEP_MICROS)
    
    yield ImgWithBoxes(
            img_path=img_path,
            img_width=w,
            img_height=h,
            bboxes=bboxes,
            microstamp=microstamp)



DATASET_NAME_TO_ITER_FACTORY = {
  'gopro1_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=[])),
  'gopro1_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=[])),
  'gopro1_fish_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=['FISH'])),
  'gopro1_fish_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=['FISH'])),
  'gopro1_head_test': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/test/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/test/images/',
      only_classes=['HEAD'])),
  'gopro1_head_train': (lambda:
    iter_gopro1_img_gts(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      only_classes=['HEAD'])),
}
