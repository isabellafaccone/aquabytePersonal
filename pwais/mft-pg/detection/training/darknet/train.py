import tempfile
import os
import mlflow
import click

from mtf_utils import misc as mft_misc


def convert_to_yolo_format(
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

  import csv

  with open(in_csv_path, newline='') as f:
    rows = list(csv.DictReader(f))

  print('Read %s rows from %s' % (len(rows), in_csv_path))

  print("Saving to %s" % out_yolo_dir)
  mft_misc.mkdir(out_yolo_dir)

  # Read and cache the image path only once
  h = None
  w = None

  train_txt_lines = []
  for row in rows:

    img_fname = os.path.basename(row['image_f'])
    img_path = os.path.join(imgs_basedir, img_fname)
    img_dest_path = os.path.join(out_yolo_dir, img_fname)
    mft_misc.run_cmd("ln -s %s %s || true" % (img_path, img_dest_path))

    train_txt_lines.append(img_dest_path)

    if (h, w) == (None, None):
      import imageio

      img = imageio.imread(img_path)
      h, w = img.shape[:2]
      
      h = float(h)
      w = float(w)
      print("Images have dimensions width %s height %s" % (w, h))
    
    # LOL those annotations aren't JSONs, they're string-ified python dicts :(
    import ast
    annos_raw = ast.literal_eval(row['annotation'])
    yolo_label_lines = []
    for anno in annos_raw['annotations']:
      if anno['category'] != positive_class:
        continue
      
      if 'xCrop' not in anno:
        print('bad anno %s' % (row,))
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
    print('saved', dest)

  with open(out_train_txt_path, 'w') as f:
    f.write('\n'.join(train_txt_lines))
  print('saved', out_train_txt_path)

  with open(out_names_path, 'w') as f:
    f.write('bkg')            # Class 0
    f.write(positive_class)   # Class 1
  print('saved', out_names_path)

  with open(out_data_path, 'w') as f:
    train_fname = os.path.basename(out_train_txt_path)
    names_fname = os.path.basename(out_names_path)
    f.write('classes=2')
    f.write('train=' + train_fname)
    f.write('valid=' + train_fname)  # IDK if this does anything
    f.write('eval=' + train_fname)   # I think this makes it compute mAP on the training set?
    f.write('names=' + names_fname)  # I think it needs this for debug images
    # f.write('backup='+ )
  print('saved', out_data_path)

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
    config_txt.replace('width=416', 'width=' + str(model_params['width']))
  if 'height' in model_params:
    config_txt.replace('height=416', 'height=' + str(model_params['height']))
  if 'max_batches' in model_params:
    config_txt.replace('max_batches = 500200', 'max_batches=' + str(model_params['max_batches']))

  return config_txt


def install_dataset(mlflow, model_workdir, dataset_name):
  if dataset_name.startswith('gopro1'):
    if 'fish' in dataset_name:
      positive_class = 'FISH'
    else:
      positive_class = 'HEAD'
    
    cparams = dict(
      in_csv_path='/opt/mft-pg/datasets/datasets_s3/gopro1/train/gopro_fish_head_anns.csv',
      imgs_basedir='/opt/mft-pg/datasets/datasets_s3/gopro1/train/images/',
      out_yolo_dir=os.path.join(model_workdir, 'gopro_fish_head_anns.csv.yolov3.annos'),
      out_train_txt_path=os.path.join(model_workdir, 'train.txt'),
      out_names_path=os.path.join(model_workdir, 'names.names'),
      out_data_path=os.path.join(model_workdir, 'data.data'),
      positive_class=positive_class,
    )
    for k, v in cparams.items():
      mlflow.log_param('convert_to_yolo_format.' + k, v)
    convert_to_yolo_format(**cparams)

  else:
    raise ValueError("Don't know how to train on %s" % dataset_name)

def install_model_config(
      mlflow,
      model_workdir,
      **model_params):
  
  template_path = '/opt/mft-pg/detection/models/detection_models_s3/yolo_ragnarok_config_hack0/yolov3-fish.cfg'
  mlflow.log_param('template_path', template_path)
  config_txt = create_model_config_str(template_path, **model_params)

  # We want to name this file 'yolov3.cfg' to make ONNX conversion easier
  # at inference time
  dest = os.path.join(model_workdir, 'yolov3.cfg')
  with open(dest, 'w') as f:
    f.write(config_txt)
  
  if model_params.get('finetune_from_imagenet'):
    mft_misc.run_cmd(
      "cd {model_workdir} && wget {weights_url}".format(
        model_workdir=model_workdir,

        # From https://github.com/AlexeyAB/darknet/tree/Yolo_v3#pre-trained-models-for-different-cfg-files-can-be-downloaded-from-smaller---faster--lower-quality
        # Maybe this is mscoco actually? IDK
        weights_url='https://pjreddie.com/media/files/darknet53.conv.74',
      ))

  

def run_darknet_training(mlflow, model_workdir):
  
  pretrained_weights_path = ''
  if os.path.exists(os.path.join(model_workdir, 'darknet53.conv.74')):
    pretrained_weights_path = 'darknet53.conv.74'
  mlflow.log_param('pretrained_weights_path', pretrained_weights_path)

  cmd = """
    cd {model_workdir} &&
      darknet detector train data.data yolov3.cfg {pretrained_weights_path} -map -dont_show test.mp4  > train_log.txt 2>&1
    """.format(
      model_workdir=model_workdir,
      pretrained_weights_path=pretrained_weights_path)

  mlflow.log_param('train_cmd', cmd)
  mft_misc.run_cmd(cmd)

  mlflow.log_artifact(os.path.join(model_workdir, 'chart.png'))
  mlflow.log_artifact(os.path.join(model_workdir, 'train_log.txt'))
  mlflow.log_artifact(os.path.join(model_workdir, 'yolov3_last.weights'))


@click.command(help="Train a Yolo fish (or fish head) detector using Darknet")
@click.option("--scratch_dir", default="/tmp")
@click.option("--dataset_name", default="gopro1_fish")
@click.option("--width", default=416)
@click.option("--height", default=416)
@click.option("--max_batches", default=30000)
@click.option("--finetune_from_imagenet", default=True)
def train_darknet_mlflow(
      scratch_dir,
      dataset_name,
      width,
      height,
      max_batches,
      finetune_from_imagenet):

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
      max_batches=max_batches,
      finetune_from_imagenet=finetune_from_imagenet)
    
    run_darknet_training(mlflow, model_workdir)

if __name__ == "__main__":
  train_darknet_mlflow()