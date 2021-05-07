
import click
import mlflow

def create_trt_from_darknet_yolo(
      yolo_config_path="",
      yolo_weights_path="",
      yolo_num_categories=2,
      out_yolo_onnx_path="",
      out_trt_engine_path=""):

  # parse the conf and get the width height and do yolov3-WxH.cfg
  # yolo_to_onnx.py --category_num={yolo_num_categories} -m {yolo_conf_path}
  # python3 /opt/tensorrt_demos/yolo/yolo_to_onnx.py --category_num=2 -m yolov3-416

  # rename the weights to match {yolo_conf_path} !
  # onnx_to_tensorrt.py -v --category_num={yolo_num_categories} -m {yolo_conf_path}
  # optional do --int8 for Xavier

  # optional do smoke test?
  # note: we should move our MyTrtYOLO to the mft lib so we can use it all over

  pass


@click.command(
  help="Create a TRT engine for the current host GPU and the given model")
@click.option("--scratch_dir", default="/tmp")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
@click.option("--save_to", default="", 
  help="Leave blank to write to a tempfile & log via mlflow")
def create_trt_engine(
      scratch_dir,
      use_model_run_id,
      use_model_artifact_dir,
      save_to):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')
  
  assert use_model_artifact_dir, "Need some model artifacts to TensorRT-ize"




  # assert detect_on_dataset in DATASET_NAME_TO_ITER_FACTORY, \
  #   "Requested %s but only have %s" % (
  #     detect_on_dataset, DATASET_NAME_TO_ITER_FACTORY.keys())

  # iter_factory = DATASET_NAME_TO_ITER_FACTORY[detect_on_dataset]
  # iter_img_gts = iter_factory()

  # if detect_limit >= 0:
  #   import itertools
  #   iter_img_gts = itertools.islice(iter_img_gts, detect_limit)

  # detector_runner = create_runner_from_artifacts(use_model_artifact_dir)
  
  # if gpu_id > 0:
  #   import os
  #   os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

  # with mlflow.start_run() as mlrun:
  #   mlflow.log_param('parent_run_id', use_model_run_id)

  #   import time
  #   print('running detector ...')
  #   start = time.time()
  #   df = detector_runner(iter_img_gts)
  #   d_time = time.time() - start
  #   print('done running in', d_time)
    
  #   mlflow.log_metric('mean_latency_ms', 1e3 * df['latency_sec'].mean())

  #   mlflow.log_metric('total_detection_time_sec', d_time)

  #   if save_to:
  #     df.to_pickle(save_to)
  #   else:
  #     import tempfile
  #     fname = str(mlrun.info.run_id) + '_detections_df.pkl'
  #     save_to = os.path.join(tempfile.gettempdir(), fname)
  #     df.to_pickle(save_to)
  #     mlflow.log_artifact(save_to)
  #   print('saved', save_to)

if __name__ == "__main__":
  create_trt_engine()

