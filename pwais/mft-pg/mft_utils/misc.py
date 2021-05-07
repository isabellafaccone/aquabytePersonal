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
    log.info("... done ({:.2f} sec) with %s " % (cmd, duration))
  return out

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
