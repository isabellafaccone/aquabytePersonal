import os

import click
import mlflow

from mft_utils import misc as mft_misc



@click.command(help="Using a tracking fixture(s), create an evaluation report(s)")
@click.option("--use_model_run_id", default="",
  help="Use the model with this mlflow run ID (optional)")
@click.option("--use_model_artifact_dir", default="",
  help="Use the model artifacts at this directory path (optional)")
def create_eval_report(
      use_model_run_id,
      use_model_artifact_dir):
  
  if use_model_run_id and not use_model_artifact_dir:
    run = mlflow.get_run(use_model_run_id)
    use_model_artifact_dir = run.info.artifact_uri
    assert 'file://' in use_model_artifact_dir, \
      "Only support local artifacts for now ..."
    use_model_artifact_dir = use_model_artifact_dir.replace('file://', '')
  
  assert use_model_artifact_dir, "Need some model artifacts to run a model"

  run_id = use_model_run_id or None
  with mlflow.start_run(run_id=run_id) as mlrun:
    mlflow.log_param('parent_run_id', use_model_run_id)

    for fname in os.listdir(use_model_artifact_dir):
      if not fname.endswith('.tracks_df.pkl'):
        continue
        
      t_path = os.path.join(use_model_artifact_dir, fname)
      mft_misc.log.info("Using tracks %s" % t_path)

      from mft_utils import df_util
      df = df_util.read_obj_df(t_path)
      
      from mft_utils import tracking_eval as t_eval
      html = t_eval.tracks_df_to_html(df)
      dest_path = t_path + '.eval.html'
      with open(dest_path, 'w') as f:
        f.write(html)
      mft_misc.log.info("Saved report to %s" % dest_path)


if __name__ == "__main__":
  create_eval_report()
