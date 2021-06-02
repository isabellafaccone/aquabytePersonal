import pandas as pd

"""

sample image at top
core stuff
latency distribution
(future: ram distribution)


(for each hist with examples, prolly wanna link it ....)
per-image average scores histogram with examples

precisions histogram with examples

recalls histogram with examples

for each optional attribute:
  histogram with examples

"""


def get_core_description_html(df):

  def _safe_get_col(colname):
    if colname in df.columns:
      return df[colname][0]
    else:
      return "(unknown)"

  core_metrics = {
    'Detector Runner': _safe_get_col('detector_runner'),
    'Detector Info': _safe_get_col('detector_info'),
    
    'Model Artifact Dir': _safe_get_col('model_artifact_dir'),
    'MLFlow Run ID': _safe_get_col('model_run_id'),
    
    'Dataset': _safe_get_col('dataset'),
    'Number of images': len(df),

    'COCO Average Precision @ 1 (0.5 IoU)': _safe_get_col('coco_overall_AP1_iou05'),
    'COCO Average Recall @ 1 (0.5 IoU)': _safe_get_col('coco_overall_AR1_iou05'),
    
    'COCO Average Precision (Small)': _safe_get_col('coco_APsmall'),
    'COCO Average Precision (Medium)': _safe_get_col('coco_APmedium'),
    'COCO Average Precision (Large)': _safe_get_col('coco_APlarge'),

    'COCO Average Recall (Small)': _safe_get_col('coco_ARsmall'),
    'COCO Average Recall (Medium)': _safe_get_col('coco_ARmedium'),
    'COCO Average Recall (Large)': _safe_get_col('coco_ARlarge'),
  }
    
  cdf = pd.DataFrame([core_metrics])
  return cdf.T.style.render()



def get_sample_row_html(df, row_idx=-1):
  from mft_utils.img_w_boxes import ImgWithBoxes
  
  if row_idx < 0 and len(df) > 0:
    import random
    rand = random.Random(1337)
    row_idx = rand.randint(0, len(df) - 1)
  
  if len(df) > 0:
    sample_row = df.iloc[row_idx]
    obj = ImgWithBoxes.from_dict(sample_row)
    for bb in obj.bboxes_alt:
      bb.category_name = "GT:" + bb.category_name

    sample_row_html = obj.to_html()
  else:
    sample_row_html = "(no data)"
  
  return sample_row_html


def get_time_series_report_html(df):
  def get_count_distinct(timescale):
    return (df['microstamp'] // int(timescale)).nunique()

  ts_metrics = {
    'Num Distinct Timestamps': get_count_distinct(1),
    'Num Distinct Minutes': get_count_distinct(60 * 1e6),
    'Num Distinct Hours': get_count_distinct(60 * 60 * 1e6),
    'Num Distinct Days': get_count_distinct(24 * 60 * 60 * 1e6),
    'Frames per second': (
      float(len(df['microstamp'])) / 
        (1e-6 * (df['microstamp'].max() - df['microstamp'].min()))),
  }
    
  tdf = pd.DataFrame([ts_metrics])
  return tdf.T.style.render()

def spark_df_add_mean_bbox_score(
        spark_df,
        bbox_col='bboxes',
        outcol='mean_bbox_score'):
  from pyspark.sql.functions import udf
  from pyspark.sql.types import DoubleType

  def mean_bbox_score(bboxes):
    import numpy as np
    return float(np.mean([bb.score for bb in bboxes] or [-1.]))
  mean_bbox_score_udf = udf(lambda x: mean_bbox_score(x), DoubleType())

  spark_df = spark_df.withColumn(outcol, mean_bbox_score_udf(bbox_col))
  return spark_df


def spark_df_maybe_add_extra_float(
        spark_df,
        key='my_key',
        default_value=-1.,
        outcol=''):
  from pyspark.sql.functions import udf
  from pyspark.sql.types import DoubleType

  if not outcol:
    outcol = key

  has_extra_key = key in spark_df.select('extra').first().extra
  if not has_extra_key:
    return spark_df

  def get_extra(extra):
    return float(extra.get(key, default_value))
  mean_bbox_score_udf = udf(lambda x: get_extra(x), DoubleType())

  spark_df = spark_df.withColumn(outcol, mean_bbox_score_udf(spark_df.extra))
  return spark_df


def get_latency_report_html(df):
  import numpy as np
  
  def get_latency_report(latencies_ms, title):
    hist, edges = np.histogram(latencies_ms, density=False, bins=100)

    from bokeh.plotting import figure

    fig = figure(
            title="Latency Distribution (milliseconds)",
            y_axis_label="Count",
            x_axis_label="Latency (milliseconds)")
    fig.quad(
        top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="blue", line_color="navy", alpha=0.85)
  
    from mft_utils.plotting import bokeh_fig_to_html
    fig_html = bokeh_fig_to_html(fig, title=title)

    stats_df = pd.DataFrame([{
      'mean': np.mean(latencies_ms),
      'median': np.percentile(latencies_ms, 50),
      '90th': np.percentile(latencies_ms, 90),
      '99th': np.percentile(latencies_ms, 99),
    }])
    stats_html = stats_df.T.style.render()

    return "<b>%s</b><br/>%s<br/>%s" % (title, fig_html, stats_html)

  reports = []
  if 'latency_sec' in df.columns and len(df) > 0:
    latencies_ms = 1e3 * df['latency_sec']
    
    reports.append(get_latency_report(
      latencies_ms, "Detector Latencies"))
  else:
    reports.append("<i>No detector latency data!!</i><br />")

  if len(df) > 0:
    REPORTS_TO_MINE = {
      'extra.preprocessor.CLAHE.latency': 'CLAHE Preprocess Latency'
    }
    for key, report_title in REPORTS_TO_MINE.items():
      if key in df.columns:
        latencies_ms = 1e3 * df[key]
      elif key.replace('extra.', '') in df['extra'][0]:
        key = key.replace('extra.', '')
        latencies_ms = 1e3 * np.array([
          float(extra[key]) for extra in df['extra']])
      reports.append(
          get_latency_report(latencies_ms, report_title))

  return "<br/>".join(reports)



def get_histogram_with_examples_htmls(df, hist_cols=[], spark=None):
  from mft_utils import misc as mft_misc

  if not hist_cols:
    hist_cols = [
      'img_coco_metrics_APrecision1_iou05',
      'img_coco_metrics_Recall1_iou05',
      'meta:mean_bbox_score',
      'meta:extra:akpd.blurriness',
      'meta:extra:akpd.darkness',
      'meta:extra:akpd.quality',
      'meta:extra:akpd.mean_luminance',
    ]

  mft_misc.log.info("Histogram-with-examples-ifying cols: %s" % (hist_cols,))

  from oarphpy import spark as S
  class MFTSpark(S.SessionFactory):
    CONF_KV = {
      'spark.files.overwrite': 'true',
        # Make it easy to have multiple invocations of parent function
    }
    #   'spark.pyspark.python': 'python3',
    #   'spark.driverEnv.PYTHONPATH': '/opt/mft-pg:/opt/oarphpy',
    # }

  from oarphpy import plotting as opl
  class Plotter(opl.HistogramWithExamplesPlotter):
    NUM_BINS = 20
    ROWS_TO_DISPLAY_PER_BUCKET = 5

    def display_bucket(self, sub_pivot, bucket_id, irows):
      # Sample from irows using reservior sampling
      import random
      rand = random.Random(1337)
      rows = []
      for i, row in enumerate(irows):
        r = rand.randint(0, i)
        if r < self.ROWS_TO_DISPLAY_PER_BUCKET:
          if i < self.ROWS_TO_DISPLAY_PER_BUCKET:
            rows.insert(r, row)
          else:
            rows[r] = row
      
      # Deserialize collected rows
      from oarphpy import spark as S
      rows = [
        S.RowAdapter.from_row(r) for r in rows
      ]

      # Now render each row to HTML
      from mft_utils.img_w_boxes import ImgWithBoxes
      row_htmls = []
      for row in rows:
        obj = ImgWithBoxes.from_dict(row)
        for bb in obj.bboxes_alt:
          bb.category_name = "GT:" + bb.category_name
        row_htmls.append(obj.to_html())
      
      HTML = """
      <b>Pivot: {spv} Bucket: {bucket_id} </b> <br/>
      
      {row_bodies}
      """.format(
            spv=sub_pivot,
            bucket_id=bucket_id,
            row_bodies="<br/><br/><br/>".join(row_htmls))
      
      return bucket_id, HTML


  col_to_html = {}
  with MFTSpark.sess(spark) as spark:
    import pyspark.sql
    if not isinstance(df, pyspark.sql.DataFrame):
      spark_rows = [
        S.RowAdapter.to_row(r) for r in df.to_dict(orient='records')
      ]
      schema = S.RowAdapter.to_schema(df.to_dict(orient='records')[0])
      df = spark.createDataFrame(spark_rows, schema=schema)
    
    df = df.persist()
    plotter = Plotter()
    for hist_col in hist_cols:
      if hist_col.startswith('meta:'):
        if hist_col == 'meta:mean_bbox_score':
          df = spark_df_add_mean_bbox_score(df)
          df = df.persist()
          hist_col = 'mean_bbox_score'
        elif hist_col.startswith('meta:extra:'):
          key = hist_col.replace('meta:extra:', '')
          outcol = key.replace('.', '_')
          df = spark_df_maybe_add_extra_float(df, key=key, outcol=outcol)
          df = df.persist()
          hist_col = outcol
      
      if hist_col not in df.columns:
        mft_misc.log.info("Skipping col: %s" % hist_col)
        continue

      fig = plotter.run(df, hist_col)

      from mft_utils.plotting import bokeh_fig_to_html
      fig_html = bokeh_fig_to_html(fig, title=hist_col)

      col_to_html[hist_col] = fig_html
  return col_to_html

    # # sdf = ks.DataFrame(
    # #   [S.RowAdapter.to_row(r) for r in df.to_dict(orient='records')])

    
    

    # from mft_utils.plotting import bokeh_fig_to_html
    # fig_html = bokeh_fig_to_html(fig, title='img_coco_metrics_APrecision1_iou05')

    # total_html = html_yay + "<br/><br/>" +cdf.T.to_html() + "<br/><br/>" + fig_html 
    # with open('/opt/mft-pg/yaytest.html', 'w') as f:
    #   f.write(total_html)
    # return total_html

def detections_df_to_html(df):

  sample_html = get_sample_row_html(df)

  core_desc_html = get_core_description_html(df)

  latency_html = get_latency_report_html(df)

  time_series_html = get_time_series_report_html(df)

  hist_col_to_html = get_histogram_with_examples_htmls(df)

  hist_agg_html = "<br/><br/>".join(
    """
      <h2>%s</h2><br/>
      <div width='90%%' height='1000px' style='border-style:dotted;overflow-y:auto'>%s</div>
    """ % (k, v)
    for k, v in hist_col_to_html.items())

  return """
    {sample_html}<br/><br/>

    {core_desc_html}<br/><br/>

    <h2>Latencies</h2><br/>
    {latency_html}
    <br/><br/>

    <h2>Time Series Info</h2>
    {time_series_html}
    <br/><br/>

    <br/><br/>
    
    <h1>Histograms with Examples</h1><br/>
    {hist_agg_html}
    """.format(
      sample_html=sample_html,
      core_desc_html=core_desc_html,
      latency_html=latency_html,
      time_series_html=time_series_html,
      hist_agg_html=hist_agg_html)

