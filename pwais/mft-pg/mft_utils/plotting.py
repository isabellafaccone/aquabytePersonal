import hashlib
import colorsys
import copy

import cv2
import imageio
import numpy as np

from mft_utils.bbox2d import BBox2D



def hash_to_rbg(x, s=0.8, v=0.8):
  """Given some value `x` (integral types work best), hash `x`
  to an `(r, g, b)` color tuple using a hue based on the hash
  and the given `s` (saturation) and `v` (lightness).
  
  Based upon OarphPy: https://github.com/pwais/oarphpy/blob/eac675c403f7c7ba61240e990bef7efced215119/oarphpy/plotting.py#L20  
  """

  # NB: ideally we just use __hash__(), but as of Python 3 it's not stable,
  # so we use a trick based upon the Knuth hash
  
  h_i = int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)
  h = (h_i % 2654435769) / 2654435769.
  rgb = 255 * np.array(colorsys.hsv_to_rgb(h, s, v))
  return tuple(rgb.astype(int).tolist())


def color_to_opencv(color):
  r, g, b = np.clip(color, 0, 255).astype(int).tolist()
  return r, g, b


def contrasting_color(color):
  r, g, b = (np.array(color) / 255.).tolist()
  
  h, s, v = colorsys.rgb_to_hsv(r, g, b)
  
  # Pick contrasting hue and lightness
  h = abs(1. - h)
  v = abs(1. - v)
  
  rgb = 255 * np.array(colorsys.hsv_to_rgb(h, s, v))  
  return tuple(rgb.astype(int).tolist())


def draw_bbox_in_image(np_image, bbox, color=None, label_txt='', thickness=2):
  """Draw a bounding box in `np_image`.
  Args:
    np_image (numpy.ndarray): Draw in this image.
    bbox: A (x1, y1, x2, y2) tuple or a bbox instance.
    color (tuple): An (r, g, b) tuple specifying the border color; by
        default use a category-determined color.
    label_txt (str): Override for the label text drawn for this box.  Prefer
        `bbox.category_name`, then this category string.  Omit label if 
        either is empty.
    thickness (int): thickness of the line in pixels.
        use the `category` attribute; omit label text if either is empty
  
  Based upon PSegs: https://github.com/pwais/psegs/blob/d24fac31a9be8d43ae0de51d6a6ca807b3c36379/psegs/util/plotting.py#L37
  """

  bbox = copy.deepcopy(bbox)
  if not isinstance(bbox, BBox2D):
    bbox = BBox2D.from_x1_y1_x2_y2(*bbox)

  label_txt = label_txt or bbox.category_name
  if not color:
    color = hash_to_rbg(label_txt)

  bbox.quantize()
  x1, y1, x2, y2 = bbox.get_x1_y1_x2_y2()

  ### Draw Box
  cv2.rectangle(
    np_image,
    (x1, y1),
    (x2, y2),
    color_to_opencv(color),
    thickness=int(thickness))

  ### Draw Text
  FONT_SCALE = 0.8
  FONT = cv2.FONT_HERSHEY_PLAIN
  PADDING = 2 # In pixels

  ret = cv2.getTextSize(label_txt, FONT, fontScale=FONT_SCALE, thickness=1)
  ((text_width, text_height), _) = ret

  # Draw the label above the bbox by default ...
  tx1, ty1 = bbox.x, bbox.y - PADDING

  # ... unless the text would draw off the edge of the image ...
  if ty1 - text_height - PADDING <= 0:
    ty1 += bbox.height + text_height + 2 * PADDING
  ty2 = ty1 - text_height - PADDING

  # ... and also shift left if necessary.
  if tx1 + text_width > np_image.shape[1]:
    tx1 -= (tx1 + text_width + PADDING - np_image.shape[1])
  tx2 = tx1 + text_width
  
  cv2.rectangle(
    np_image,
    (tx1, ty1 + PADDING),
    (tx2, ty2 - PADDING),
    color_to_opencv(color),
    cv2.FILLED)

  text_color = contrasting_color(color)
  cv2.putText(
    np_image,
    label_txt,
    (tx1, ty1),
    FONT,
    FONT_SCALE,
    color_to_opencv(text_color),
    1) # thickness


class BBoxVideoRecorder(object):
  def __init__(self, outpath='/tmp/bboxe_video.mp4', fps=10):
    self._outpath = outpath
    self._writer = imageio.get_writer(outpath, fps=fps)

  def record(self, img, bboxes, identify_by='category_name'):
    img = img.copy()
    for bbox in bboxes:
      bbox.draw_in_image(img, identify_by=identify_by)
    self._writer.append_data(img)
  
  def close(self):
    self._writer.close()
  
  def __del__(self):
    self.close()

def bokeh_fig_to_html(fig, title=''):
  # Yes sadly it seems Bokeh has no "write to buffer" option
  import tempfile
  from oarphpy.plotting import save_bokeh_fig
  with tempfile.NamedTemporaryFile() as f:
    save_bokeh_fig(fig, f.name, title=title)
    return f.read().decode("utf-8")
  
  # We tried this and some of the javascript didn't work, might have been
  # a bad version of bokeh tho.
  # from bokeh.resources import CDN
  # from bokeh.embed import file_html
  # html = file_html(fig, CDN, title)
  # return html


def get_latency_report(latencies_ms, title):
  import numpy as np
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

  import pandas as pd
  stats_df = pd.DataFrame([{
    'mean': np.mean(latencies_ms),
    'median': np.percentile(latencies_ms, 50),
    '90th': np.percentile(latencies_ms, 90),
    '99th': np.percentile(latencies_ms, 99),
  }])
  stats_html = stats_df.T.style.render()

  return "<b>%s</b><br/>%s<br/>%s" % (title, fig_html, stats_html)

def gen_corr_plot_html(
        xs, ys,
        slope=1.,
        intercept=0.,
        x_title='',
        y_title='',
        stats_kv={}):
  
  from bokeh.plotting import figure

  title = f"Correlation report for {y_title} vs {x_title}"
  fig = figure(
          title=title,
          y_axis_label=y_title,
          x_axis_label=x_title)
  fig.circle(xs, ys, color='orange')

  from bokeh.models import Slope
  line = Slope(gradient=slope, y_intercept=intercept, line_color='blue')
  fig.add_layout(line)

  from mft_utils.plotting import bokeh_fig_to_html
  fig_html = bokeh_fig_to_html(fig, title=title)

  import pandas as pd
  stats = dict(stats_kv)
  stats.update({
    'Coefficient': slope,
    'Intercept': intercept,
  })
  stats_df = pd.DataFrame([stats])
  stats_html = stats_df.T.style.render()

  return f"{fig_html}<br/ >{stats_html}"
