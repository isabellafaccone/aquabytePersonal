
###############################################################################
## Preprocessor Impls

def CLAHE_enhance(
        img,
        clip_limit=5,
        tile_grid_size=(8, 8),
        sharpen=True,
        sharp_grid_size=(21, 21),
        sharpen_weight=2.0):
  """Based upon Aquabyte research-lib CLAHE:
      * https://github.com/aquabyte-new/internal-tools/blob/7544ddc3f1c74e4b198d8bbe1984b41c414ef680/research-lib/src/research_lib/utils/picture.py#L24 

  Ported here to reduce dependencies and work with MFT-PG images (RGB)
  """

  import cv2

  # Convert to LAB
  image_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

  l_channel, a_channel, b_channel = cv2.split(image_lab)
  
  # apply CLAHE to lightness channel
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  cl = clahe.apply(l_channel)

  # merge the CLAHE enhanced L channel with the original A and B channel
  merged_channels = cv2.merge((cl, a_channel, b_channel))

  # convert image from LAB color model back to RGB color model
  enhanced_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

  if sharpen:
    blurred = cv2.GaussianBlur(enhanced_image, sharp_grid_size, 0)
    enhanced_image = cv2.addWeighted(enhanced_image, sharpen_weight, blurred, 1-sharpen_weight, 0)

  return enhanced_image


###############################################################################
## Preprocessor Interface and Hooks

class PreprocessorRunner(object):
  def __init__(self):
    self._preprocessors = []

  @classmethod
  def build_from_configs(cls, preprocessor_configs):
    runner = cls()
    for p_conf in preprocessor_configs:
      if p_conf == 'clahe':
        runner._preprocessors.append(CLAHE())
      else:
        raise ValueError(
          "Don't know how to create preproc %s of %s" % (
            p_conf, preprocessor_configs))
    return runner

  def preprocess(self, img):
    import time
    
    pp_to_stats = {}
    for p in self._preprocessors:  
      start = time.time()
      img = p(img)
      p_time = time.time() - start
      pp_to_stats[p.get_name()] = p_time
    return img, pp_to_stats


class PreprocessorBase(object):
  def __call__(self, img):
    return img
  
  @classmethod
  def get_name(cls):
    return str(cls.__name__)

class CLAHE(PreprocessorBase):
  def __call__(self, img):
    return CLAHE_enhance(img)