"""
mlflow run --no-conda . --entry-point=train_yolo_zoo

"""

from concurrent.futures import ThreadPoolExecutor

import attr
import click
import mlflow

def get_num_gpus():
  import pycuda.autoinit
  import pycuda.driver as cuda
  return cuda.Device.count()

@attr.s()
class TrainTask(object):
  train_yolo_darknet_params = attr.ib(default={})
  skip_eval = attr.ib(default=False)
  test_dataset_name = attr.ib(default='')

  def run(self):
    with mlflow.start_run(nested=True) as train_run:
      mlflow.projects.run(
          run_id=train_run.info.run_id,
          uri=".",
          entry_point="train_yolo_darknet",
          parameters=self.train_yolo_darknet_params,
          use_conda=False)

      # Why do we have to log these again ..?
      mlflow.log_params(self.train_yolo_darknet_params)
      # succeeded = p.wait()
      # assert succeeded

      train_run_id = train_run.info.run_id
    
    if not self.skip_eval:
      # with mlflow.start_run(nested=True) as test_run:
        mlflow.projects.run(
            run_id=train_run.info.run_id,
            uri=".",
            entry_point="create_detection_fixture",
            parameters=dict(
              use_model_run_id=train_run_id,
              detect_on_dataset=self.test_dataset_name,
              gpu_id=self.train_yolo_darknet_params.get('gpu_id', -1),
            ),
            use_conda=False)
        # succeeded = p.wait()
        # assert succeeded



@click.command(help="Train a Zoo of yolo models")
@click.option("--scratch_dir", default="/tmp")
@click.option("--train_dataset_name", default="gopro1_fish_train")
@click.option("--test_dataset_name", default="gopro1_fish_test")
@click.option("--min_width", default=16*10)
@click.option("--max_width", default=16*65)
@click.option("--width_step", default=16*2)
@click.option("--skip_eval", default=False)
@click.option("--max_batches", default=100)
@click.option("--finetune_from_imagenet", default=True)
@click.option("--leave_gpu0_free", default=True)
@click.option("--parallel", default=True)
def train_yolo_zoo(
      scratch_dir,
      train_dataset_name,
      test_dataset_name,
      min_width,
      max_width,
      width_step,
      skip_eval,
      max_batches,
      finetune_from_imagenet,
      leave_gpu0_free,
      parallel):
  
  # We will round-robin assign GPUS
  n_workers = 1
  if parallel:
    n_gpus = get_num_gpus()
    gpu_ids = list(range(n_gpus))
    if leave_gpu0_free:
      gpu_ids = [gid for gid in gpu_ids if gid != 0]
    n_workers = len(gpu_ids)
  else:
    gpu_ids = [-1]
  import itertools
  import six
  iter_gpu_ids = itertools.cycle(gpu_ids)

  # Create the model-training tasks
  tasks = []
  for width in range(min_width, max_width + width_step, width_step):
    tasks.append(
      TrainTask(
        train_yolo_darknet_params=dict(
          scratch_dir=scratch_dir,
          dataset_name=train_dataset_name,
          input_width=width,
          input_height=width, # For now we just do one aspect ratio
          max_batches=max_batches,
          finetune_from_imagenet=finetune_from_imagenet,
          gpu_id=six.next(iter_gpu_ids),
        ),
        skip_eval=skip_eval,
        test_dataset_name=test_dataset_name,
    ))

  TAGS = {
    'mft-trainer': 'train_yolo_zoo',
  }
  with mlflow.start_run(tags=TAGS) as mlrun:
  
    import time
    start = time.time()

    print('num tasks', len(tasks))
    # x = [t.run() for t in tasks]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
      _ = executor.map(lambda t: t.run(), tasks)

    mlflow.log_metric('total_zoo_time', time.time() - start)

if __name__ == "__main__":
  train_yolo_zoo()