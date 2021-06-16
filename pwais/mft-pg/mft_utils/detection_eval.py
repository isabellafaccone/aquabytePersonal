import copy

import pandas as pd

from mft_utils import plotting as mft_plotting



def to_spark_df(spark, pdf):
  import pyspark.sql
  if isinstance(pdf, pyspark.sql.DataFrame):
    return pdf

  import copy
  from oarphpy import spark as S
  spark_rows = [
    S.RowAdapter.to_row(r) for r in pdf.to_dict(orient='records')
  ]
  schema_row = copy.deepcopy(pdf.to_dict(orient='records')[0])
  if not schema_row['bboxes']:
    for row in pdf.to_dict(orient='records'):
      if row['bboxes']:
        schema_row['bboxes'] = copy.deepcopy(row['bboxes'])
        break
    assert schema_row['bboxes'], \
      'hacks! need row prototype and sniffing failed'
  if not schema_row['bboxes_alt']:
    for row in pdf.to_dict(orient='records'):
      if row['bboxes_alt']:
        schema_row['bboxes_alt'] = copy.deepcopy(row['bboxes_alt'])
        break
    assert schema_row['bboxes_alt'], \
      'hacks! need row prototype and sniffing failed'
  schema = S.RowAdapter.to_schema(schema_row)
  df = spark.createDataFrame(spark_rows, schema=schema)
  return df


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
    obj = sample_row['img_bb']
    for bb in obj.bboxes_alt:
      bb.category_name = "GT:" + bb.category_name

    sample_row_html = obj.to_html()
  else:
    sample_row_html = "(no data)"
  
  return sample_row_html




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

def spark_df_add_num_bboxes(spark_df,
        bbox_col='bboxes',
        outcol='num_bboxes'):
  
  from pyspark.sql import functions as F
  spark_df = spark_df.withColumn(outcol, F.size(spark_df[bbox_col]))
  return spark_df

def spark_df_maybe_add_extra_float(
        spark_df,
        key='my_key',
        outcol=''):

  from pyspark.sql.functions import udf
  from pyspark.sql.types import DoubleType

  if not outcol:
    outcol = key

  has_extra_key = key in spark_df.select('extra').first().extra
  if not has_extra_key:
    return spark_df

  vals = spark_df.select(spark_df['extra'][key].cast('float').alias(outcol))
  num_non_nas = vals.na.drop().count()
  if num_non_nas == 0:
    return spark_df

  spark_df = spark_df.withColumn(outcol, spark_df['extra'][key].cast('float'))

  return spark_df

def spark_df_maybe_add_postprocessor_score(
        spark_df,
        name='',
        outcol=''):

    if not outcol:
      outcol = name
    
    postproc_key = name
    postproc_extract_score = lambda v: float('nan')
    if name in ('max_SAO_score', 'min_SAO_score'):
      postproc_key = 'SAOScorer'
      def get_fish_scores(res):
        if not res.fish_clusters:
          return [float('nan')]
        else:
          return [
            float(c[0].extra['SAO_score'])
            for c in res.fish_clusters
          ]
      if name == 'max_SAO_score':
        postproc_extract_score = lambda res: max(get_fish_scores(res))
      else:
        postproc_extract_score = lambda res: min(get_fish_scores(res))


    sample_row = spark_df.select('postprocessor_to_result').first()[0]
    if postproc_key in sample_row.keys():
      from pyspark.sql.functions import udf

      def get_pp_result_as_Row(pp_to_result):
        result_pkl, stats = pp_to_result[postproc_key]
        
        from mft_utils.img_bbox_postprocessing import decode_postproc_result
        result = decode_postproc_result(result_pkl)

        from oarphpy.spark import RowAdapter
        return RowAdapter.to_row(result)

      from oarphpy.spark import RowAdapter
      schema = RowAdapter.to_schema(get_pp_result_as_Row(sample_row))

      get_pp_result_as_Row_udf = udf(lambda x: get_pp_result_as_Row(x), schema)

      result_col_name = 'postproc_result_' + postproc_key
      spark_df = spark_df.withColumn(
                  result_col_name,
                  get_pp_result_as_Row_udf(spark_df['postprocessor_to_result']))
      
      from pyspark.sql.types import DoubleType
      score_result_udf = udf(lambda x: postproc_extract_score(x), DoubleType())
      spark_df = spark_df.withColumn(
                  outcol,
                  score_result_udf(spark_df[result_col_name]))

    return spark_df



def get_latency_report_html(df):
  import numpy as np

  reports = []
  if 'detector_latency_sec' in df.columns and len(df) > 0:
    latencies_ms = 1e3 * df['detector_latency_sec']
    
    reports.append(mft_plotting.get_latency_report(
      latencies_ms, "Detector Latencies"))
  else:
    reports.append("<i>No detector latency data!!</i><br />")

  if len(df) > 0:
    REPORTS_TO_MINE = {
      'extra.preprocessor.CLAHE.latency': 'CLAHE Preprocess Latency',
    }

    img_bb = df.to_dict(orient='records')[0]['img_bb']
    for postproc_key in img_bb.postprocessor_to_result.keys():
      REPORTS_TO_MINE['postproc.' + postproc_key] = (
        "Postprocessor Latency: %s" % postproc_key)


    for key, report_title in REPORTS_TO_MINE.items():
      if key in df.columns:
        latencies_ms = 1e3 * df[key]
        reports.append(
          mft_plotting.get_latency_report(latencies_ms, report_title))
      elif key.replace('extra.', '') in df['extra'][0]:
        key = key.replace('extra.', '')
        latencies_ms = 1e3 * np.array([
          float(extra[key]) for extra in df['extra']])
        reports.append(
          mft_plotting.get_latency_report(latencies_ms, report_title))
      elif key.startswith('postproc.'):
        postproc_key = key.replace('postproc.', '')
        latencies_ms = []
        for pp_to_result in df['postprocessor_to_result']:
          entry = pp_to_result[postproc_key]
          result_pkl, p_time_sec = entry
          latencies_ms.append(1e3 * p_time_sec)
        latencies_ms = np.array(latencies_ms)
        reports.append(
          mft_plotting.get_latency_report(latencies_ms, report_title))

  return "<br/>".join(reports)



def get_histogram_with_examples_htmls(df, hist_cols=[], spark=None):
  from mft_utils import misc as mft_misc

  if not hist_cols:
    hist_cols = [
      'meta:extra:coco_metrics_APrecision1_iou05',
      'meta:extra:coco_metrics_Recall1_iou05',
      'meta:num_bboxes',
      'meta:mean_bbox_score',
      'meta:postprocessor:max_SAO_score',
      'meta:postprocessor:min_SAO_score',
      'meta:extra:akpd.blurriness',
      'meta:extra:akpd.darkness',
      'meta:extra:akpd.quality',
      'meta:extra:akpd.mean_luminance',
      'meta:extra:akpd_synth.num_fish_with_occluded',
    ]

  mft_misc.log.info("Histogram-with-examples-ifying cols: %s" % (hist_cols,))

  from oarphpy import spark as S
  class MFTSpark(S.SessionFactory):
    CONF_KV = {
      'spark.files.overwrite': 'true',
        # Make it easy to have multiple invocations of parent function
      'spark.driver.memory': '16g',
    }

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
      for img_bb in rows:

        pp_result_key = None
        for key in row.asDict().keys():
          if key.startswith('postproc_result_'):
            pp_result_key = key.replace('postproc_result_', '')
        
        if pp_result_key:
          pp_result = img_bb.get_postprocessor_result(pp_result_key)
          row_htmls.append(pp_result.to_html(debug_img_src=img_bb))
        else:
          for bb in img_bb.bboxes_alt:
            bb.category_name = "GT:" + bb.category_name
          row_htmls.append(img_bb.to_html())
      
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
    df = to_spark_df(spark, df)
    
    base_df = df.repartition('img_path').persist()
    plotter = Plotter()
    for hist_col in hist_cols:
      df = base_df
      if hist_col.startswith('meta:'):
        if hist_col == 'meta:mean_bbox_score':
          df = spark_df_add_mean_bbox_score(df)
          df = df.persist()
          hist_col = 'mean_bbox_score'
        elif hist_col == 'meta:num_bboxes':
          df = spark_df_add_num_bboxes(df)
          df = df.persist()
          hist_col = 'num_bboxes'
        elif hist_col.startswith('meta:extra:'):
          key = hist_col.replace('meta:extra:', '')
          outcol = key.replace('.', '_')
          df = spark_df_maybe_add_extra_float(df, key=key, outcol=outcol)
          df = df.persist()
          hist_col = outcol
        elif hist_col.startswith('meta:postprocessor:'):
          key = hist_col.replace('meta:postprocessor:', '')
          outcol = key.replace('.', '_')
          df = spark_df_maybe_add_postprocessor_score(
                    df, name=key, outcol=outcol)
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

def get_bbox_histogram_with_examples_htmls(df, bbox_miners=[], spark=None):
  from mft_utils import misc as mft_misc

  if not bbox_miners:
    bbox_miners = [
      'meta:score',
      'meta:extra:SAO_score',
      'alt:meta:extra:frac_px_visible',
    ]
  
  mft_misc.log.info(
    "Histogram-with-examples BBox Miners: %s" % (bbox_miners,))

  from oarphpy import plotting as opl
  class BBoxPlotter(opl.HistogramWithExamplesPlotter):
    NUM_BINS = 20
    ROWS_TO_DISPLAY_PER_BUCKET = 10

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

      # Now render each row to HTML
      from mft_utils.img_w_boxes import ImgWithBoxes
      from mft_utils.bbox2d import BBox2D
      row_htmls = []
      for row in rows:
        from oarphpy import spark as S
        bbox = S.RowAdapter.from_row(row['bbox'])
        img_w_boxes = S.RowAdapter.from_row(row['img_bb'])
        row_htmls.append(bbox.to_html(debug_img_src=img_w_boxes))
      
      HTML = """
      <b>Pivot: {spv} Bucket: {bucket_id} </b> <br/>
      
      {row_bodies}
      """.format(
            spv=sub_pivot,
            bucket_id=bucket_id,
            row_bodies="<br/><br/><br/>".join(row_htmls))
      
      return bucket_id, HTML




  from oarphpy import spark as S
  class MFTSpark(S.SessionFactory):
    CONF_KV = {
      'spark.files.overwrite': 'true',
        # Make it easy to have multiple invocations of parent function
      'spark.driver.memory': '24g',
    }

  miner_to_html = {}
  with MFTSpark.sess(spark) as spark:
    orig_df = to_spark_df(spark, df)
    orig_df = orig_df.repartition(
                orig_df.rdd.getNumPartitions() * 50,
                'bboxes')
    orig_df = orig_df.persist()
    
    plotter = BBoxPlotter()
    for hist_col in bbox_miners:
      df = orig_df

      if hist_col.startswith('alt:'):
        box_col = 'bboxes_alt'
        hist_col = hist_col.replace('alt:', '')
      else:
        box_col = 'bboxes'

      from pyspark.sql import functions as F
      df = df.withColumn('bbox', F.explode(box_col))
      df = df.withColumn('extra', df['bbox.extra'])
      df = df.withColumn('bbox_detector_score', df['bbox.score'])
      df = df.persist()

      if hist_col.startswith('meta:'):
        if hist_col == 'meta:score':
          hist_col = 'bbox_detector_score'
        elif hist_col.startswith('meta:extra:'):
          key = hist_col.replace('meta:extra:', '')
          outcol = key.replace('.', '_')
          
          df = spark_df_maybe_add_extra_float(df, key=key, outcol=outcol)

          df = df.persist()
          hist_col = outcol
      
      if hist_col not in df.columns:
        mft_misc.log.info("Skipping col / miner: %s" % hist_col)
        continue

      fig = plotter.run(df, hist_col)

      from mft_utils.plotting import bokeh_fig_to_html
      fig_html = bokeh_fig_to_html(fig, title=hist_col)

      miner_to_html[hist_col] = fig_html
  return miner_to_html



def detections_df_to_html(df):

  sample_html = get_sample_row_html(df)

  core_desc_html = get_core_description_html(df)

  latency_html = get_latency_report_html(df)

  # time_series_html = get_time_series_report_html(df)

  hist_col_to_html = get_histogram_with_examples_htmls(df)

  bbox_report_to_html = get_bbox_histogram_with_examples_htmls(df)

  hist_agg_html = "<br/><br/>".join(
    """
      <h2>%s</h2><br/>
      <div width='90%%' height='1000px' style='resize:both;border-style:inset;border-width: 5px;overflow-y:auto'>%s</div>
    """ % (k, v)
    for k, v in hist_col_to_html.items())

  bbox_agg_html = "<br/><br/>".join(
    """
      <h2>%s</h2><br/>
      <div width='90%%' height='1000px' style='resize:both;border-style:dashed double none;border-width: 5px;overflow-y:auto'>%s</div>
    """ % (k, v)
    for k, v in bbox_report_to_html.items())

  return """
    {sample_html}<br/><br/>

    {core_desc_html}<br/><br/>

    <h2>Latencies</h2><br/>
    {latency_html}
    <br/><br/>
    
    <h1>Images: Histograms with Examples</h1><br/>
    {hist_agg_html}
    <br/><br/>

    <h1>BBoxes: Histograms with Examples</h1><br/>
    {bbox_agg_html}

    """.format(
      sample_html=sample_html,
      core_desc_html=core_desc_html,
      latency_html=latency_html,
      # time_series_html=time_series_html,
      hist_agg_html=hist_agg_html,
      bbox_agg_html=bbox_agg_html)

