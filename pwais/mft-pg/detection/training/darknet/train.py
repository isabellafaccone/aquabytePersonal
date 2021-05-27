import os

import mlflow
import click

from mft_utils import misc as mft_misc


PRETRAINED_WEIGHTS_DEFAULT_PATH = '/tmp/darknet53.conv.74'


def convert_csv_to_yolo_format(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      out_yolo_dir='gopro_fish_head_anns.csv.yolov3.annos',
      out_train_txt_path='train.txt',
      out_names_path='names.names',
      out_data_path='data.data',
      positive_class='FISH'):

  # https://pjreddie.com/darknet/yolo/
  # "Darknet wants a .txt file for each image with a line for each ground truth
  #   object in the image that looks like:
  #   <object-class> <x> <y> <width> <height>
  #   Where x, y, width, and height are relative to the image's
  #   width and height."

  # TODO: sync this code with the more modern:
  # gopro_data.iter_gopro1_img_gts()

  import csv

  with open(in_csv_path, newline='') as f:
    rows = list(csv.DictReader(f))

  mft_misc.log.info('Read %s rows from %s' % (len(rows), in_csv_path))

  mft_misc.log.info("Saving to %s" % out_yolo_dir)
  mft_misc.mkdir(out_yolo_dir)

  # Read and cache the image dims only once; they're from a video so all
  # the same
  h = None
  w = None

  train_txt_lines = []
  for row in rows:

    img_fname = os.path.basename(row['image_f'])
    img_path = os.path.join(imgs_basedir, img_fname)
    img_dest_path = os.path.join(out_yolo_dir, img_fname)
    mft_misc.run_cmd(
      "ln -s %s %s || true" % (img_path, img_dest_path), nolog=True)

    train_txt_lines.append(img_dest_path)

    if (h, w) == (None, None):
      import imageio

      img = imageio.imread(img_path)
      h, w = img.shape[:2]
      
      h = float(h)
      w = float(w)
      mft_misc.log.info("Images have dimensions width %s height %s" % (w, h))
    
    # LOL those annotations aren't JSONs, they're string-ified python dicts :(
    import ast
    annos_raw = ast.literal_eval(row['annotation'])
    yolo_label_lines = []
    for anno in annos_raw['annotations']:
      if anno['category'] != positive_class:
        continue
      
      if 'xCrop' not in anno:
        mft_misc.log.warn('bad anno %s' % (row,))
        continue

      x_pixels = anno['xCrop']
      y_pixels = anno['yCrop']
      w_pixels = anno['width']
      h_pixels = anno['height']
      
      x_center_rel = (x_pixels + .5 * w_pixels) / w
      y_center_rel = (y_pixels + .5 * h_pixels) / h
      w_rel = w_pixels / w
      h_rel = h_pixels / h

      yolo_label_lines.append(
        '1 {x_center_rel} {y_center_rel} {w_rel} {h_rel}'.format(
          x_center_rel=x_center_rel,
          y_center_rel=y_center_rel,
          w_rel=w_rel,
          h_rel=h_rel))
    
    dest_label_fname = img_fname.replace('jpg', 'txt')
    dest = os.path.join(out_yolo_dir, dest_label_fname)
    with open(dest, 'w') as f:
      f.write('\n'.join(yolo_label_lines))
    # print('saved', dest)

  with open(out_train_txt_path, 'w') as f:
    f.write('\n'.join(train_txt_lines))
  mft_misc.log.info('saved %s' % out_train_txt_path)

  with open(out_names_path, 'w') as f:
    f.write('\n'.join((
      'bkg',          # Class 0
      positive_class  # Class 1
    )))
  mft_misc.log.info('saved %s' % out_names_path)

  with open(out_data_path, 'w') as f:
    train_fname = os.path.basename(out_train_txt_path)
    names_fname = os.path.basename(out_names_path)
    f.write('\n'.join((
      'classes=2',
      'train=' + train_fname,
      'valid=' + train_fname, # I think this makes it compute mAP on the training set?
      # 'eval=' + train_fname,  # IDK if this does anything
      'names=' + names_fname, # I think it needs this for debug images
      'backup=.',
    )))
  mft_misc.log.info('saved %s' % out_data_path)


def convert_img_gt_to_darknet_format(
      img_gts=[],
      out_yolo_dir='img_gts.yolov3.annos',
      out_train_txt_path='train.txt',
      out_valid_txt_path='valid.txt',
      out_names_path='names.names',
      out_data_path='data.data',
      id_to_category=[]):

  # FMI see convert_csv_to_yolo_format() above

  mft_misc.log.info('Saving %s img_gts to Darknet format' % (len(img_gts),))
  mft_misc.log.info("Saving to %s" % out_yolo_dir)
  mft_misc.mkdir(out_yolo_dir)

  if not id_to_category:
    import itertools
    id_to_category = sorted(set(
                      itertools.chain.from_iterable(
                        (b.category_name for b in img_gt.bboxes)
                        for img_gt in img_gts)))
  
  # # For some reason darknet insists on 2-class testing but not training
  if len(id_to_category) == 1:
    id_to_category =  ['__mft_bg__'] + id_to_category
  category_to_id = dict((c, i) for i, c in enumerate(id_to_category))

  train_txt_lines = []
  for i, img_gt in enumerate(img_gts):
    
    img_path = img_gt.img_path
    img_fname = os.path.basename(img_path)
    img_fname = 'darknet_train_%s_%s' % (i, img_fname)
    img_dest_path = os.path.join(out_yolo_dir, img_fname)
    mft_misc.run_cmd(
      "ln -s %s %s || true" % (img_path, img_dest_path), nolog=True)

    train_txt_lines.append(img_dest_path)

    h, w = (None, None)
    yolo_label_lines = []
    for bbox in img_gt.bboxes:
      x_pixels = bbox.x
      y_pixels = bbox.y
      w_pixels = bbox.width
      h_pixels = bbox.height

      assert bbox.im_width > 0, bbox.im_width
      assert bbox.im_height > 0, bbox.im_height
      
      x_center_rel = (x_pixels + .5 * w_pixels) / bbox.im_width
      y_center_rel = (y_pixels + .5 * h_pixels) / bbox.im_height
      w_rel = w_pixels / bbox.im_width
      h_rel = h_pixels / bbox.im_height

      yolo_label_lines.append(
        '{class_id} {x_center_rel} {y_center_rel} {w_rel} {h_rel}'.format(
          class_id=category_to_id[bbox.category_name],
          x_center_rel=x_center_rel,
          y_center_rel=y_center_rel,
          w_rel=w_rel,
          h_rel=h_rel))
    
    dest_label_fname = img_fname.replace('.jpg', '.txt')
    dest_label_fname = dest_label_fname.replace('.png', '.txt')
    dest = os.path.join(out_yolo_dir, dest_label_fname)
    with open(dest, 'w') as f:
      f.write('\n'.join(yolo_label_lines))

  with open(out_train_txt_path, 'w') as f:
    f.write('\n'.join(train_txt_lines))
  mft_misc.log.info('saved %s' % out_train_txt_path)

  import random
  rand = random.Random(1337)
  valid_lines = list(train_txt_lines)
  rand.shuffle(valid_lines)
  valid_lines = valid_lines[:100]
  with open(out_valid_txt_path, 'w') as f:
    f.write('\n'.join(valid_lines))
  mft_misc.log.info('saved %s' % out_valid_txt_path)

  with open(out_names_path, 'w') as f:
    f.write('\n'.join((
      id_to_category
    )))
  mft_misc.log.info('saved %s' % out_names_path)

  with open(out_data_path, 'w') as f:
    train_fname = os.path.basename(out_train_txt_path)
    valid_fname = os.path.basename(out_valid_txt_path)
    names_fname = os.path.basename(out_names_path)
    f.write('\n'.join((
      'classes=' + str(len(id_to_category)),
      'train=' + train_fname,
      'valid=' + valid_fname, # I think this makes it compute mAP on the training set?
      # 'eval=' + train_fname,  # IDK if this does anything
      'names=' + names_fname, # I think it needs this for debug images
      'backup=.',
    )))
  mft_misc.log.info('saved %s' % out_data_path)


def create_model_config_str(
      template_path='/opt/mft-pg/detection/models/detection_models_s3/yolo_ragnarok_config_hack0/yolov3-fish.cfg',
      **model_params):

  # NB: we can't use python configparser because
  # the Yolo config isn't valid according to that module--
  # there are duplicate sections e.g. for conv layers
  # So we use dumb find and replace

  with open(template_path, 'r') as f:
    config_txt = f.read()

  if 'width' in model_params:
    config_txt = config_txt.replace('width=416', 'width=' + str(model_params['width']))
  if 'height' in model_params:
    config_txt = config_txt.replace('height=416', 'height=' + str(model_params['height']))
  if 'batch_size' in model_params:
    config_txt = config_txt.replace('batch=4', 'batch=' + str(model_params['batch_size']))

    import math
    subdivisions = max(1, int(model_params['batch_size']) // 4)
    config_txt = config_txt.replace('subdivisions=1', 'subdivisions=' + str(subdivisions))
      # Save GPU memory: use small subdivisions
  if 'max_batches' in model_params:
    config_txt = config_txt.replace('max_batches = 500200', 'max_batches=' + str(model_params['max_batches']))
  if 'classes' in model_params:
    config_txt = config_txt.replace('classes=2', 'classes=' + str(model_params['classes']))

    # Guh, Yolo doesn't auto-resize ....
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    filters = (int(model_params['classes']) + 5) * 3
    config_txt = config_txt.replace('filters=21', 'filters=%s' % filters)

  return config_txt


def install_dataset(mlflow, model_workdir, dataset_name):
  if dataset_name.startswith('gopro1'):
    if 'fish' in dataset_name:
      positive_class = 'FISH'
    else:
      positive_class = 'HEAD'
    
    assert 'train' in dataset_name

    out_names_path = os.path.join(model_workdir, 'names.names')
    cparams = dict(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      out_yolo_dir=os.path.join(model_workdir, 'gopro_fish_head_anns.csv.yolov3.annos'),
      out_train_txt_path=os.path.join(model_workdir, 'train.txt'),
      out_names_path=out_names_path,
      out_data_path=os.path.join(model_workdir, 'data.data'),
      positive_class=positive_class,
    )
    for k, v in cparams.items():
      mlflow.log_param('convert_to_yolo_format.' + k, v)
    
    import time
    start = time.time()
    convert_csv_to_yolo_format(**cparams)
    mlflow.log_metric('convert_to_yolo_format_time_sec', time.time() - start)

    # Need this for inference... it's not in the model config
    mlflow.log_artifact(out_names_path)

  elif dataset_name.startswith('akpd1') or dataset_name.startswith('hrf_'):
    if dataset_name.startswith('akpd1'):
      from mft_utils.akpd_data import DATASET_NAME_TO_ITER_FACTORY
      assert dataset_name in DATASET_NAME_TO_ITER_FACTORY, (
        dataset_name, 'not in', DATASET_NAME_TO_ITER_FACTORY.keys())
      
      img_gts_factory = DATASET_NAME_TO_ITER_FACTORY[dataset_name]
      img_gts = list(img_gts_factory())
    elif dataset_name.startswith('hrf_'):
      from mft_utils.high_recall_fish_data import DATASET_NAME_TO_ITER_FACTORY
      assert dataset_name in DATASET_NAME_TO_ITER_FACTORY, (
        dataset_name, 'not in', DATASET_NAME_TO_ITER_FACTORY.keys())
      
      img_gts_factory = DATASET_NAME_TO_ITER_FACTORY[dataset_name]
      img_gts = list(img_gts_factory())
    else:
      raise ValueError("can't handle %s" % dataset_name)

    out_names_path = os.path.join(model_workdir, 'names.names')
    out_data_path = os.path.join(model_workdir, 'data.data')
    cparams = dict(
      img_gts=img_gts,
      out_yolo_dir=os.path.join(model_workdir, 'img_gt.yolov3.annos'),
      out_train_txt_path=os.path.join(model_workdir, 'train.txt'),
      out_valid_txt_path=os.path.join(model_workdir, 'valid.txt'),
      out_names_path=out_names_path,
      out_data_path=out_data_path,
    )
    for k, v in cparams.items():
      if k != 'img_gts':
        mlflow.log_param('convert_to_darknet_format.' + k, v)
    
    import time
    start = time.time()
    convert_img_gt_to_darknet_format(**cparams)
    mlflow.log_metric('convert_to_darknet_format_time_sec', time.time() - start)

    # Need this for inference... it's not in the model config
    mlflow.log_artifact(out_names_path)
    
    # Save this for reference
    mlflow.log_artifact(out_data_path)

  else:
    raise ValueError("Don't know how to train on %s" % dataset_name)


def install_model_config(
      mlflow,
      model_workdir,
      **model_params):
  
  template_path = '/opt/mft-pg/detection/models/detection_models_s3/yolo_ragnarok_config_hack0/yolov3-fish.cfg'
  mlflow.log_param('template_path', template_path)

  if 'classes' not in model_params:
    dat_path = os.path.join(model_workdir, 'data.data')
    category_num = mft_misc.darknet_get_yolo_category_num(dat_path)
    model_params['classes'] = str(category_num)

  config_txt = create_model_config_str(template_path, **model_params)

  # We want to name this file 'yolov3.cfg' to make ONNX conversion easier
  # at inference time
  dest = os.path.join(model_workdir, 'yolov3.cfg')
  with open(dest, 'w') as f:
    f.write(config_txt)
  mft_misc.log.info("saved model config %s" % dest)
  mlflow.log_artifact(dest)
  
  if model_params.get('finetune_from_imagenet'):
    if not os.path.exists(PRETRAINED_WEIGHTS_DEFAULT_PATH):
      mft_misc.run_cmd(
        "cd /tmp && wget {weights_url}".format(
          model_workdir=model_workdir,

          # From https://github.com/AlexeyAB/darknet/tree/Yolo_v3#pre-trained-models-for-different-cfg-files-can-be-downloaded-from-smaller---faster--lower-quality
          # Maybe this is mscoco actually? IDK
          weights_url='https://pjreddie.com/media/files/darknet53.conv.74',
        ))
    mft_misc.run_cmd(
      "ln -s %s %s" % (
        PRETRAINED_WEIGHTS_DEFAULT_PATH,
        os.path.join(model_workdir, 'darknet53.conv.74')))

  

def run_darknet_training(mlflow, model_workdir, gpu_id):
  
  pretrained_weights_path = ''
  if os.path.exists(os.path.join(model_workdir, 'darknet53.conv.74')):
    pretrained_weights_path = 'darknet53.conv.74'
  mlflow.log_param('pretrained_weights_path', pretrained_weights_path)

  cmd = """
    cd {model_workdir} &&
      {gpu_prefix} darknet detector train data.data yolov3.cfg {pretrained_weights_path} -map -dont_show test.mp4  > train_log.txt
    """.format(
      model_workdir=model_workdir,
      pretrained_weights_path=pretrained_weights_path,
      gpu_prefix="CUDA_VISIBLE_DEVICES=%s" % gpu_id if gpu_id >= 0 else "")

  mlflow.log_param('train_cmd', cmd)
  mft_misc.log.info('Starting training in %s' % model_workdir)
  import time
  start = time.time()
  mft_misc.run_cmd(cmd)
  mlflow.log_metric('training_time_sec', time.time() - start)

  mlflow.log_artifact(os.path.join(model_workdir, 'chart.png'))
  mlflow.log_artifact(os.path.join(model_workdir, 'train_log.txt'))
  mlflow.log_artifact(os.path.join(model_workdir, 'yolov3_final.weights'))


@click.command(help="Train a Yolo fish (or fish head) detector using Darknet")
@click.option("--scratch_dir", default="/tmp")
@click.option("--dataset_name", default="gopro1_fish_train")
@click.option("--width", default=416)
@click.option("--height", default=416)
@click.option("--batch_size", default=4)
@click.option("--max_batches", default=30000)
@click.option("--finetune_from_imagenet", default=True)
@click.option("--clean_scratch", default=True)
@click.option("--gpu_id", default=-1)
def train_darknet_mlflow(
      scratch_dir,
      dataset_name,
      width,
      height,
      batch_size,
      max_batches,
      finetune_from_imagenet,
      clean_scratch,
      gpu_id):

  assert os.path.exists('/opt/i_am_darknet_trainer'), \
    "This script may break if run outside of the darketn training container"

  TAGS = {
    'mft-trainer': 'darknet',
  }

  with mlflow.start_run(tags=TAGS) as mlrun:

    # Darknet dumps a ton of stuff in the cwd, so we create a scratch dir
    # for the run
    model_workdir = os.path.join(
                  scratch_dir, 'mft-pg-runs', mlrun.info.run_id)
    mft_misc.mkdir(model_workdir)
    mlflow.log_param('model_workdir', model_workdir)

    install_dataset(mlflow, model_workdir, dataset_name)
    install_model_config(
      mlflow,
      model_workdir,
      width=width,
      height=height,
      batch_size=batch_size,
      max_batches=max_batches,
      finetune_from_imagenet=finetune_from_imagenet)
    
    run_darknet_training(mlflow, model_workdir, gpu_id)
  
  if clean_scratch:
    mft_misc.run_cmd("rm -rf %s" % model_workdir)

if __name__ == "__main__":
  train_darknet_mlflow()
