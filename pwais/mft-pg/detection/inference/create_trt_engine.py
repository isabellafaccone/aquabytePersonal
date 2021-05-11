import os
from sys import exec_prefix

import click
import mlflow

from mft_utils import misc as mft_misc



def create_trt_from_darknet_yolo(
      yolo_config_path="yolov3.cfg",
      yolo_weights_path="yolov3.weights",
      out_dir="/tmp/"):

  # The Yolo conversion scripts below take in config / weights files
  # with the name pattern `yolov3-WxH.cfg` where W and H are the
  # input image size.  Rather than hack up the scripts, we'll
  # just use symlinks to organize the inputs into an expected format.
  w, h = mft_misc.darknet_get_yolo_input_wh(yolo_config_path)
  category_num = mft_misc.darknet_get_yolo_category_num(yolo_config_path)

  if 'yolov3' in yolo_config_path:
    mname = 'yolov3-%sx%s' % (w, h)
  else:
    raise ValueError("Can't handle this yolo yet: %s" % yolo_config_path)
  
  with mft_misc.error_friendly_tempdir() as tempdirpath:
    conf_path = os.path.join(tempdirpath, mname + '.cfg')
    weights_path = os.path.join(tempdirpath, mname + '.weights')
    onnx_path = os.path.join(tempdirpath, mname + '.onnx')
    trt_engine_path = os.path.join(tempdirpath, mname + '.trt')

    mft_misc.run_cmd("ln -s %s %s" % (yolo_config_path, conf_path))
    mft_misc.run_cmd("ln -s %s %s" % (yolo_weights_path, weights_path))

    # Convert darknet -> ONNX
    mft_misc.run_cmd(f"""
      cd {tempdirpath} && \
      python3 /opt/tensorrt_demos/yolo/yolo_to_onnx.py \
        --category_num={category_num} -m {mname}
    """)
    assert os.path.exists(onnx_path), onnx_path

    # Convert ONNX -> TRT
    mft_misc.run_cmd(f"""
      cd {tempdirpath} && \
      python3 /opt/tensorrt_demos/yolo/onnx_to_tensorrt.py \
        -v --category_num={category_num} -m {mname}
    """)
    assert os.path.exists(trt_engine_path), trt_engine_path
  
    out_yolo_onnx_path = os.path.join(out_dir, 'yolov3.onnx')
    out_trt_engine_path = os.path.join(
                              out_dir,
                              ('yolov3.' + 
                                mft_misc.cuda_get_device_trt_engine_name() + 
                                '.trt'))
    mft_misc.run_cmd("cp -v %s %s" % (onnx_path, out_yolo_onnx_path))
    mft_misc.run_cmd("cp -v %s %s" % (trt_engine_path, out_trt_engine_path))


  # parse the conf and get the width height and do yolov3-WxH.cfg
  # yolo_to_onnx.py --category_num={yolo_num_categories} -m {yolo_conf_path}
  # python3 /opt/tensorrt_demos/yolo/yolo_to_onnx.py --category_num=2 -m yolov3-416

  # rename the weights to match {yolo_conf_path} !
  # python3 /opt/tensorrt_demos/yolo/onnx_to_tensorrt.py -v --category_num={yolo_num_categories} -m {yolo_conf_path}
  # optional do --int8 for Xavier

  # optional do smoke test?
  # note: we should move our MyTrtYOLO to the mft lib so we can use it all over

class TRTBuilder(object):
  def __init__(self, artifact_dir):
    self._artifact_dir = artifact_dir

  def __call__(self, workdir):
    pass

class AVTTRTDemosYoloBuilder(TRTBuilder):
  def __call__(self, workdir):
    assert os.path.exists('/opt/tensorrt_demos'), \
      "This builder designed to work with https://github.com/jkjung-avt/tensorrt_demos"
    
    if not os.path.exists('/opt/tensorrt_demos/plugins/libyolo_layer.so'):
      mft_misc.log.info("Lazy-building Yolo plugin ...")
      mft_misc.run_cmd("""
        cd /opt/tensorrt_demos/plugins && make
      """)

    yolo_config_path = os.path.join(self._artifact_dir, "yolov3.cfg")
    yolo_weights_path = os.path.join(self._artifact_dir, "yolov3_final.weights")
    create_trt_from_darknet_yolo(
      yolo_config_path=yolo_config_path,
      yolo_weights_path=yolo_weights_path,
      out_dir=workdir)


def create_trt_runner_from_artifacts(artifact_dir):
  if os.path.exists(os.path.join(artifact_dir, 'yolov3_final.weights')):
    return AVTTRTDemosYoloBuilder(artifact_dir)
  else:
    raise ValueError("Could not resolve builder for %s" % artifact_dir)


@click.command(
  help="Create a TRT engine for the current host GPU and the given model")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--scratch_dir", default="/tmp")
@click.option("--clean_scratch", default=True)
def create_trt_engine(
      use_model_run_id,
      use_model_artifact_dir,
      scratch_dir,
      clean_scratch):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')

  assert use_model_artifact_dir, "Need some model artifacts to TensorRT-ize"

  workdir = os.path.join(
                  scratch_dir,
                  'mft-create_trt_engine',
                  use_model_run_id or 'anon_run')
  mft_misc.mkdir(workdir)

  trt_runner = create_trt_runner_from_artifacts(use_model_artifact_dir)

  run_id = use_model_run_id or None
  with mlflow.start_run(run_id=run_id) as mlrun:
    mlflow.log_param('trt_parent_run_id', use_model_run_id)

    import time
    mft_misc.log.info('Runing TRT runner with workspace to %s ...' % workdir)
    start = time.time()
    trt_runner(workdir)
    duration = time.time() - start

    mlflow.log_metric('total_trt_build_time_sec', duration)
    mlflow.log_artifacts(workdir)

    if clean_scratch:
      mft_misc.run_cmd("rm -rf %s" % workdir)

if __name__ == "__main__":
  create_trt_engine()

