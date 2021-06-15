import pandas as pd

from mft_utils import tracking
from mft_utils import misc as mft_misc
from mft_utils import plotting as mft_plotting

def get_debug_video_html(imbbs, parallel=-1, only_track_id=''):
  import tempfile
  with tempfile.NamedTemporaryFile(suffix='_tracks_debug.mp4') as f:
    with open(f.name, 'wb') as w:
      debug_img_kwargs = dict(tracking.BASE_DEBUG_IMG_KWARGS)
      debug_img_kwargs['only_track_id'] = only_track_id
      tracking.write_debug_video(
        w.name,
        imbbs,
        parallel=parallel,
        debug_img_kwargs=debug_img_kwargs)
    return mft_misc.mp4_video_to_inline_html(video_path=f.name)


def get_core_description_html(df):

  def _safe_get_col(colname):
    if colname in df.columns:
      return df[colname][0]
    else:
      return "(unknown)"

  core_metrics = {
    'Tracker Runner': _safe_get_col('tracker_name'),
    'Tracker Info': _safe_get_col('tracker_info'),
    'Tracker Ablated to given Frames Per Second?':
       str(_safe_get_col('tracker_ablate_input_to_fps')) + ' Frames per second',
    
    'Detections Model Artifact Dir': _safe_get_col('model_artifact_dir'),
    'Detections MLFlow Run ID': _safe_get_col('model_run_id'),
    
    'Dataset': _safe_get_col('dataset'),
    'Number of frames': len(df),
  }
    
  cdf = pd.DataFrame([core_metrics])
  return cdf.T.style.render()


def get_latency_report_html(df):

  reports = []
  if 'tracker_latency_sec' in df.columns and len(df) > 0:
    latencies_sec = df['tracker_latency_sec']
    
    # For ablation tests, the ablated frames have -1 latency
    latencies_sec = latencies_sec[latencies_sec != -1]
    latencies_ms = 1e3 * latencies_sec
    
    
    reports.append(mft_plotting.get_latency_report(
      latencies_ms, "Tracker Latencies"))
  else:
    reports.append("<i>No tracker latency data!!</i><br />")

  return "<br/>".join(reports)


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


def get_tracklet_histogram_with_examples_htmls(df, hist_cols=[], spark=None):
  from mft_utils import misc as mft_misc

  """
  treereduce:
   * (track_id, list[img_bb <-- with only the track_id?])

  to table:
  track_id | category_name | first_seen | last_seen | lifetime | min_bbox_score | max_bbox_score | img_bb_datas

  hist with examples:
   * max_score
   * lifetime
  

  """


  if not hist_cols:
    hist_cols = [
      'max_bbox_score',
      'lifetime_sec',
      
      
      # 'meta:extra:coco_metrics_APrecision1_iou05',
      # 'meta:extra:coco_metrics_Recall1_iou05',
      
      # 'meta:mean_bbox_score',
      # 'meta:postprocessor:max_SAO_score',
      # 'meta:postprocessor:min_SAO_score',
      # 'meta:extra:akpd.blurriness',
      # 'meta:extra:akpd.darkness',
      # 'meta:extra:akpd.quality',
      # 'meta:extra:akpd.mean_luminance',
      # 'meta:extra:akpd_synth.num_fish_with_occluded',
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
      from mft_utils.img_w_boxes import ImgWithBoxes
      rows = [
        S.RowAdapter.from_row(r) for r in rows
      ]

      # Now render each row to HTML
      row_htmls = []
      for row in rows:
        track_id = row.track_id
        imbbs = row.img_bbs

        debug_video_html = get_debug_video_html(
                              imbbs,
                              only_track_id=track_id,
                              parallel=1)

        d = row.asDict(recursive=False)
        tracklet_info = dict(
                          (k, d[k]) for k in (
                            'track_id',
                            'category_name',
                            'first_seen',
                            'last_seen', 
                            'mean_score',
                          ))

        import pandas as pd
        import pandas as pd
        info_df = pd.DataFrame([tracklet_info])
        info_html = info_df.T.style.render()

        row_html = """
        <hr>
        {debug_video_html}
        <br/>
        {info_html}
        """.format(debug_video_html=debug_video_html, info_html=info_html)
        row_htmls.append(row_html)
      
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

    img_bbs = list(df['img_bb'])
    img_bb_rdd = spark.sparkContext.parallelize(img_bbs, numSlices=len(img_bbs))

    def to_track_imbbs(imbb):
      for bbox in imbb.bboxes:
        yield bbox.track_id, [imbb]
    
    def agg_imbbs(ii1, ii2):
      return ii1 + ii2

    ti_rdd = img_bb_rdd.flatMap(to_track_imbbs)

    import pyspark
    ti_rdd = ti_rdd.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    tis_rdd = ti_rdd.reduceByKey(agg_imbbs)
    
    def to_row(tis):
      track_id = tis[0]
      img_bbs = tis[1]

      # It's easier to compute some columns here versus using the DataFrame API
      # (which is probably faster)
      category_name = ''
      bb_scores = []
      first_seen = float('inf')
      last_seen = -1
      for imbb in img_bbs:
        first_seen = min(first_seen, imbb.microstamp)
        last_seen = max(last_seen, imbb.microstamp)
        for bbox in imbb.bboxes:
          if bbox.track_id == track_id:
            category_name = bbox.category_name
            bb_scores.append(bbox.score)
      
      import numpy as np
      bb_scores = np.array(bb_scores)


      from pyspark.sql import Row
      row = Row(
        track_id=track_id,
        category_name=category_name,
        first_seen=first_seen,
        last_seen=last_seen,
        min_score=bb_scores.min(),
        max_score=bb_scores.max(),
        mean_score=np.mean(bb_scores),
        lifetime_sec=1e-6 * (last_seen - first_seen),
        n_frames=len(img_bbs),
        img_bbs=img_bbs)
      from oarphpy.spark import RowAdapter
      return RowAdapter.to_row(row)
    
    row_rdd = tis_rdd.map(to_row)
    df = spark.createDataFrame(row_rdd, samplingRatio=1)

    
    base_df = df.repartition('track_id').persist()
    plotter = Plotter()
    for hist_col in hist_cols:
      df = base_df

      # if hist_col.startswith('meta:'):
      #   if hist_col == 'meta:max_bbox_score':
      #     df = df.withColumn('max_bbox_score', F.max(df[''])

      #     df = spark_df_add_mean_bbox_score(df)
      #     df = df.persist()
      #     hist_col = 'mean_bbox_score'
      #   elif hist_col == 'meta:num_bboxes':
      #     df = spark_df_add_num_bboxes(df)
      #     df = df.persist()
      #     hist_col = 'num_bboxes'
      #   elif hist_col.startswith('meta:extra:'):
      #     key = hist_col.replace('meta:extra:', '')
      #     outcol = key.replace('.', '_')
      #     df = spark_df_maybe_add_extra_float(df, key=key, outcol=outcol)
      #     df = df.persist()
      #     hist_col = outcol
      #   elif hist_col.startswith('meta:postprocessor:'):
      #     key = hist_col.replace('meta:postprocessor:', '')
      #     outcol = key.replace('.', '_')
      #     df = spark_df_maybe_add_postprocessor_score(
      #               df, name=key, outcol=outcol)
      #     df = df.persist()
      #     hist_col = outcol
      
      if hist_col not in df.columns:
        mft_misc.log.info("Skipping col: %s" % hist_col)
        continue

      fig = plotter.run(df, hist_col)

      from mft_utils.plotting import bokeh_fig_to_html
      fig_html = bokeh_fig_to_html(fig, title=hist_col)

      col_to_html[hist_col] = fig_html
  return col_to_html


def tracks_df_to_html(df):


  """

  debug video at top; show the ground truths if ablation?

  tracklet mining:
   * overall: tracklet score
   * overall: tracklet lifetime
  
   * tracker latencies!! also detector latencies maybe
   * time_series_html

  time series:
   * CDF of fish per (minute, hour) vs score -- can we bucket by CDF cutoff?
   * number of fish by minute -- bucket by minute

  if ablation:
     * show MOTS metrics

  """

  debug_video_html = get_debug_video_html(list(df['img_bb']))

  core_desc_html = get_core_description_html(df)

  latency_html = get_latency_report_html(df)

  time_series_html = get_time_series_report_html(df)

  hist_col_to_html = get_tracklet_histogram_with_examples_htmls(df)

  hist_agg_html = "<br/><br/>".join(
    """
      <h2>%s</h2><br/>
      <div width='90%%' height='1000px' style='resize:both;border-style:inset;border-width: 5px;overflow-y:auto'>%s</div>
    """ % (k, v)
    for k, v in hist_col_to_html.items())

  return """
    {debug_video_html}<br/><br/>

    {core_desc_html}<br/><br/>

    <h2>Latencies</h2><br/>
    {latency_html}
    <br/><br/>

    <h2>Time Series Info</h2>
    {time_series_html}
    <br/><br/>

    <br/><br/>
    
    <h1>Tracklets: Histograms with Examples</h1><br/>
    {hist_agg_html}

    """.format(
      debug_video_html=debug_video_html,
      core_desc_html=core_desc_html,
      latency_html=latency_html,
      time_series_html=time_series_html,
      hist_agg_html=hist_agg_html)
