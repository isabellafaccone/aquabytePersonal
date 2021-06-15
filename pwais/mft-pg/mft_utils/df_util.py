
import pandas as pd

def df_add_static_col(df, colname, v, empty_v=None):
  import six
  if empty_v is None:
    if hasattr(v, 'items'):
      empty_v = {}
    elif isinstance(v, list):
      empty_v = []
    elif isinstance(v, six.string_types):
      empty_v = ''
    else:
      empty_v = v
  df.insert(0, colname, [v] + ([empty_v] * max(0, len(df) - 1)))

def to_obj_df(v):
  from oarphpy.spark import RowAdapter
  if isinstance(v, pd.DataFrame):
    rows = [RowAdapter.to_row(d) for d in v.to_dict(orient='records')]
  else:
    rows = [RowAdapter.to_row(o).asDict() for o in v]

  df = pd.DataFrame(rows)
  return df

def read_obj_df(path):
  from oarphpy.spark import RowAdapter
  from pyspark.sql import Row
  df = pd.read_pickle(path)
  
  from mft_utils.img_w_boxes import ImgWithBoxes
  from mft_utils.bbox2d import BBox2D
  imbbs = [RowAdapter.from_row(Row(**d)) for d in df.to_dict(orient='records')]
  df.insert(0, 'img_bb', imbbs)
  return df
