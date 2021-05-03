import os

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

def run_cmd(cmd, collect=False):
  import subprocess
  cmd = cmd.replace('\n', '').strip()
  if collect:
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
  else:
    subprocess.check_call(cmd, shell=True)
    out = None
  return out