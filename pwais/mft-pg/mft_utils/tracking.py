
import time

from mft_utils import misc as mft_misc
  

BASE_DEBUG_IMG_KWARGS = dict(
  identify_by='track_id', 
  only_track_id='',
  show_alt=True,
  alt_identify_by='category_name',
)

def write_debug_video(
        outpath,
        imbbs,
        video_height=500,
        fps=-1,
        parallel=-1,
        debug_img_kwargs=BASE_DEBUG_IMG_KWARGS):
  
  if not imbbs:
    return None
  
  if fps < 0:
    import numpy as np
    microstamps = np.array([i.microstamp for i in imbbs])
    periods_sec = 1e-6 * np.abs(microstamps[:-1] - microstamps[1:])
    avg_period = np.mean(periods_sec)
    fps = 1. / avg_period
  
  def get_t_debug_frame(idx):
    imbb = imbbs[idx]
    debug_img = imbb.get_debug_image(**debug_img_kwargs)

    import cv2
    h, w = debug_img.shape[:2]
    scale = video_height / float(h)
    target_width = int(scale * w)
    debug_img = cv2.resize(debug_img, (target_width, video_height))
    return imbb.microstamp, debug_img

  n_tasks = len(imbbs)
  iter_t_debug = mft_misc.futures_threadpool_safe_pmap(
                    get_t_debug_frame,
                    range(n_tasks),
                    parallel=parallel)
  
  def iter_in_order(iter_t_debug, n_buffer=100):
    import queue
    pq = queue.PriorityQueue()
    
    def pop():
      head_t, head_debug_image = pq.get()
      pq.task_done()
      return head_debug_image

    for i, (t, debug_image) in enumerate(iter_t_debug):
      pq.put((t, debug_image))
      if i > n_buffer:
        yield pop()
    while not pq.empty():
      yield pop()
  iter_debugs = iter_in_order(iter_t_debug)

  import imageio
  writer = imageio.get_writer(outpath, fps=fps)
  mft_misc.log.info('Generating tracker debug video ...')
  for i, debug_img in enumerate(iter_debugs):
    writer.append_data(debug_img)
    if ((i+1) % 100) == 0:
      mft_misc.log.info(
        '... generated %s of %s frames to %s' % (
          i+1, n_tasks, outpath))
  writer.close()


class MOTrackersTracker(object):

  def __init__(
        self,
        tracker_type='SORT',
        tracker_kwargs={},
        class_name_to_id={},
        include_eval_output=True):
    
    self._class_name_to_id = class_name_to_id
    self._include_eval_output = include_eval_output
    
    if tracker_type == 'SORT':
      from motrackers import SORT
      if 'max_lost' not in tracker_kwargs:
        tracker_kwargs['max_lost'] = 3
      if 'iou_threshold' not in tracker_kwargs:
        tracker_kwargs['iou_threshold'] = 0.3
      self._tracker = SORT(**tracker_kwargs)
    else:
      raise ValueError("Don't know how to create %s" % tracker_type)
    
    self.tracker_type = tracker_type
    self.tracke_params = tracker_kwargs

  def update_and_set_tracks(self, img_bb):
    bboxes = img_bb.bboxes

    bbox_coords_to_bbox = {}
    confidences = []
    class_ids = []
    t_bboxes = []
    for bbox in bboxes:
      if bbox.tracker_ignore:
        continue

      # motrackers does not make it easy to "join" bbox output with bbox input,
      # so we do the "hack" of joining input and output by bbox coords.  Note
      # that motrackers takes bboxes with integer coordinates so we use that as
      # the coordinate "join key".
      t_bbox = [
        int(coord) for coord in (bbox.x, bbox.y, bbox.width, bbox.height)]
      t_bboxes.append(t_bbox)
      bbox_coords_to_bbox[tuple(t_bbox)] = bbox

      confidences.append(bbox.score)

      # Dynamically grow category ID map if needed
      if bbox.category_name not in self._class_name_to_id:
        last_id = -1
        if self._class_name_to_id:
          last_id = max(self._class_name_to_id.values())
        self._class_name_to_id[bbox.category_name] = last_id + 1
      class_ids.append(self._class_name_to_id[bbox.category_name])

    start = time.time()
    import numpy as np
    t_bboxes = np.array(t_bboxes)
    tracks = self._tracker.update(t_bboxes, confidences, class_ids)
    update_time = time.time() - start
    img_bb.tracker_latency_sec = update_time

    for track in tracks:
      # frame_id = track[0]
      track_id = track[1]
      track_bb_key = (track[2], track[3], track[4], track[5])
      confidence = track[6]

      bbox = bbox_coords_to_bbox[track_bb_key]
      bbox.track_id = str(track_id)
      bbox.extra['tracker_confidence'] = str(confidence)

      if self._include_eval_output:
        output = ','.join(str(v) for v in track)
        bbox.extra['tracker.mot_challenge_output'] = output
