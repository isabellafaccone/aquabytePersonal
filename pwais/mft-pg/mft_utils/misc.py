import os
import sys
import time
from contextlib import contextmanager

_LOGS = {}
def create_log(name='mft'):
  global _LOGS
  if name not in _LOGS:
    import logging
    LOG_FORMAT = "%(asctime)s\t%(name)-4s %(process)d : %(message)s"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    log.addHandler(console_handler)
    _LOGS[name] = log
  return _LOGS[name]

# NB: Spark workers will lazy-construct and cache logger instances
log = create_log()

def mkdir(path):
  # From oarphpy https://github.com/pwais/oarphpy/blob/da6b87cd1ed6b68189078c560ce13c6e290eb88f/oarphpy/util/misc.py#L423
  import errno
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

def run_cmd(cmd, collect=False, nolog=False):
  import subprocess
  cmd = cmd.replace('\n', '').strip()
  if not nolog:
    log.info("Running %s ..." % cmd)
  
  start = time.time()
  if collect:
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
  else:
    subprocess.check_call(cmd, shell=True)
    out = None
  duration = time.time() - start

  if (not nolog) and (duration > 1):
    log.info("... done ({:.2f} sec) with {} ".format(duration, cmd))
  return out


def cuda_get_device_trt_engine_name(gpu_id=0):
  import pycuda.autoinit
  import pycuda.driver as cuda
  full_name = cuda.Device(gpu_id).name()
  full_name = full_name.replace(' ', '.')
  return full_name


@contextmanager
def error_friendly_tempdir(**kwargs):
  """Like `tempfile.TemporaryDirectory()`, but don't delete the tempdir
  if there's an oops.  See also: https://bugs.python.org/issue36422
  """
  import tempfile
  dirpath = tempfile.mkdtemp(**kwargs)
  try:
    yield dirpath
  finally:
    import shutil
    shutil.rmtree(dirpath)


def futures_threadpool_safe_pmap(f, iargs, parallel=-1, threadpool_kwargs={}):
  from concurrent.futures import ThreadPoolExecutor
  from concurrent.futures import as_completed

  if not 'max_workers' in threadpool_kwargs:
    if parallel < 0:
      parallel = os.cpu_count()
    threadpool_kwargs['max_workers'] = parallel

  futures = []
  with ThreadPoolExecutor(**threadpool_kwargs) as executor:
    for arg in iargs:
      futures.append(executor.submit(f, arg))
    for future in as_completed(futures):
      yield future.result()

      
def foreach_threadpool_safe_pmap(f, iargs, parallel=-1, threadpool_kwargs={}):
  return list(futures_threadpool_safe_pmap(
    f, iargs, parallel=parallel, threadpool_kwargs=threadpool_kwargs))



def _get_group_0(re_s, s):
  import re
  try:
    return re.search(re_s, s).groups()[0]
  except Exception as e:
    raise Exception("%s %s %s" % (re_s, s, e))

def darknet_get_yolo_input_wh(yolo_config_path):
  w, h = (None, None)
  with open(yolo_config_path) as f:
    for line in f.readlines():
      if w is None and 'width' in line:
        w = int(_get_group_0(r"width\W?=\W?(\d+)", line))
      if h is None and 'height' in line:
        h = int(_get_group_0(r"height\W?=\W?(\d+)", line))
  return w, h

def darknet_get_yolo_category_num(yolo_config_path):
  with open(yolo_config_path) as f:
    for line in f.readlines():
      if 'classes' in line:
        category_num = int(_get_group_0(r"classes\W?=\W?(\d+)", line))
        return category_num


def download_from_s3(uris, dest_root, parallel=-1):
  """A quick and dirty S3 downloader that only depends on `awscli`; designed
  to run outside an MFT-PG dockerized environment."""

  log.info("Have %s urls to download" % len(uris))
  
  def download(uri):
    try:
      from urlparse import urlparse
    except ImportError:
      from urllib.parse import urlparse
    
    res = urlparse(uri, allow_fragments=False)
    dest_relpath = res.path.lstrip('/')
    dest = os.path.join(dest_root, dest_relpath)

    dest_parent = os.path.split(dest)[0]
    mkdir(dest_parent)

    cmd = "aws s3 cp %s %s" % (uri, dest)
    run_cmd(cmd) 

  max_workers = os.cpu_count() * 2 if parallel < 0 else parallel
  foreach_threadpool_safe_pmap(
    download,
    uris,
    threadpool_kwargs={'max_workers': max_workers})
