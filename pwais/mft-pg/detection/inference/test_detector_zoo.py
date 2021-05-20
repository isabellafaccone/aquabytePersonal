"""
mlflow run --no-conda . --entry-point=test_detector_zoo

"""

import click
import mlflow

from mft_utils import misc as mft_misc

# The first Yolo zoo didn't have run ID tracking set up right.  Here are the
# MLFlow Run Ids for that zoo
FIRST_YOLO_ZOO_RUN_IDS = (
  '90fafa4fb2194c70ab5e99bf94964587',
  '3771d3357f534c689a31f7278f2fe60e',
  '86d0fbfae7d9432bb15b17113bf3f291',
  '058730c05f8f4dd8a3299fb10dece255',
  '32779f411bbd4c1dafe483ca8a636601',
  'e10cae25a3884f75919c2b6212e4824f',
  '3b33e4f6bc45466c9a1d35ee839a0c75',
  'b18e9351ef4b4da9b788cc135339f457',
  'e235ff91d5c043979f31b8ce92c1e7a5',
  'c114c649c02349389b54810c526bdc7f',
  '95a388e4809d4e24870e4d147140c356',
  '95d86a448acc4d86ba2d3714795453a2',
  'ea283b03af834744a616fa740b4f303a',
  'f231ff83d1dd494fa6269916cc0b1ab3',
  '1ab6658a60a7402ba0398cf445ba22d2',
  '2de468aed4e445a6a352db4b15fd703c',
  '71058625298a4b99a91f5099fd8c7cd4',
  '2176e2dd712949c5bb7e1c25abc3b278',
  '4ff330ca02054a1db9eff79abb06841a',
  '3a10b05bc3344922b0b953f27e61b07b',
  'd7e6b25641e34edd913c66d0e6725a6b',
  'f0af1e307acc4d19b13b89ab3025ccf6',
  '81fe7f4ff374481198a0aa1e94ebe3a1',
  '10a236b73ea64e84b0f7628fa3c6f0f6',
  'e4297a980b7e43ac8099454601895c0a',
  '0053813569944f2ca919993ed3bea4f9',
  '5f7c3d1b1e0c49e1a52b25ec0bf6d316',
  '4e3232e14ae6472a86717be9ce8572a7',
)


@click.command(help="Create detection fixtures for an entire detector zoo")
@click.option("--use_train_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--eval_trt_only", is_flag=True,
  help="Only build and evaluate TensorRT engine")
@click.option("--use_existing_trt", is_flag=True,
  help="Use TensorRT engines that have already been built (if available)")
def test_detector_zoo(
      use_train_run_id,
      eval_trt_only,
      use_existing_trt):
  
  run_ids_to_test = []
  if use_train_run_id:
    run_ids_to_test = [use_train_run_id]

  if not run_ids_to_test:
    run_ids_to_test = FIRST_YOLO_ZOO_RUN_IDS
  
  with mlflow.start_run(nested=True) as test_zoo_run:

    for i, run_id in enumerate(run_ids_to_test):
      mft_misc.log.info("Testing %s" % run_id)

      if not eval_trt_only:
        mlflow.projects.run(
              run_id=run_id,
              uri=".",
              entry_point="create_detection_fixture",
              parameters={
                'use_model_run_id': run_id,
              },
              use_conda=False)

      if not use_existing_trt:
        mlflow.projects.run(
            run_id=run_id,
            uri=".",
            entry_point="create_trt_engine",
            parameters={
              'use_model_run_id': run_id,
            },
            use_conda=False)
      
      # This will now grab the TRT engine
      mlflow.projects.run(
          run_id=run_id,
          uri=".",
          entry_point="create_detection_fixture",
          parameters={
            'use_model_run_id': run_id,
          },
          use_conda=False)

    mft_misc.log.info("Tested %s of %s runs" % (i+1, len(run_ids_to_test)))

if __name__ == "__main__":
  test_detector_zoo()
