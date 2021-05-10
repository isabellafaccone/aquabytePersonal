import typing

import attr

# Based on psegs https://github.com/pwais/psegs/blob/be13b8c69a36c3f9e258ab03b5a22b87f23baa3b/psegs/datum/bbox2d.py#L21
@attr.s(slots=True, eq=True, weakref_slot=False)
class BBox2D(object):
  
  # NB: We explicitly disable `validator`s so that the user may temporarily
  # use floats

  x = attr.ib(type=int, default=0, validator=None)
  """int: Base x coordinate in pixels."""
  
  y = attr.ib(type=int, default=0, validator=None)
  """int: Base y coordinate in pixels."""
  
  # perhaps eventually: depth_meters = attr.ib()

  width = attr.ib(type=int, default=0, validator=None)
  """int: Width of box in pixels."""

  height = attr.ib(type=int, default=0, validator=None)
  """int: Height of box in pixels."""

  im_width = attr.ib(type=int, default=0, validator=None)
  """int, optional: Width of enclosing image"""

  im_height = attr.ib(type=int, default=0, validator=None)
  """int, optional: Height of enclosing image"""
  
  category_name = attr.ib(type=str, default="")
  """str, optional: Class associated with this bounding box."""

  track_id = attr.ib(type=str, default="")
  """str, optional: Track ID associated with this bounding box."""

  score = attr.ib(type=float, default=0)
  """float, optional: A score associated with this box."""

  extra = attr.ib(default={}, type=typing.Dict[str, str])
  """Dict[str, str]: A map for adhoc extra context"""

  def update(self, **kwargs):
    """Update attributes of this `BBox2D` as specified in `kwargs`"""
    for k in self.__slots__:
      if k in kwargs:
        setattr(self, k, kwargs[k])

  @staticmethod
  def of_size(width, height):
    """Create a `BBox2D` of `width` by `height`"""
    return BBox2D(
            x=0, y=0,
            width=width, height=height,
            im_width=width, im_height=height)

  @staticmethod
  def from_x1_y1_x2_y2(x1, y1, x2, y2):
    """Create a `BBox2D` from corners `(x1, y1)` and `(x2, y2)` (inclusive)"""
    b = BBox2D()
    b.set_x1_y1_x2_y2(x1, y1, x2, y2)
    return b

  def set_x1_y1_x2_y2(self, x1, y1, x2, y2):
    """Update this `BBox2D` to have corners `(x1, y1)` and `(x2, y2)`
    (inclusive)"""
    self.update(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)

  def get_x1_y1_x2_y2(self):
    """Get the corners `(x1, y1)` and `(x2, y2)` (inclusive) of this `BBox2D`"""
    return self.x, self.y, self.x + self.width - 1, self.y + self.height - 1

  def get_r1_c1_r2_r2(self):
    """Get the row-major corners `(y1, x1)` and `(y2, x2)` (inclusive) of
    this `BBox2D`"""
    return self.y, self.x, self.y + self.height - 1, self.x + self.width - 1

  def get_x1_y1(self):
    """Return the origin"""
    return self.x, self.y

  def get_fractional_xmin_ymin_xmax_ymax(self, clip=True):
    """Get the corners `(x1, y1)` and `(x2, y2)` (inclusive) of this
    `BBox2D` in image-relative coordinates; i.e. each corner is scaled
    to [0, 1] based upon image size.  Forbid off-image corners only if
    `clip`."""
    xmin = float(self.x) / self.im_width
    ymin = float(self.y) / self.im_height
    xmax = float(self.x + self.width) / self.im_width
    ymax = float(self.y + self.height) / self.im_height
    if clip:
      xmin, ymin, xmax, ymax = \
        map(lambda x: float(np.clip(x, 0, 1)), \
          (xmin, ymin, xmax, ymax))
    return xmin, ymin, xmax, ymax

  def add_padding(self, *args):
    """Extrude this `BBox2D` with the given padding: either a single value
    in pixels or a `(pad_x, pad_y)` tuple."""
    if len(args) == 1:
      px, py = args[0], args[0]
    elif len(args) == 2:
      px, py = args[0], args[1]
    else:
      raise ValueError(len(args))
    self.x -= px
    self.y -= py
    self.width += 2 * px
    self.height += 2 * py

  def is_full_image(self):
    """Does this `BBox2D` cover the whole image?"""
    return (
      self.x == 0 and
      self.y == 0 and
      self.width == self.im_width and
      self.height == self.im_height)

  def get_corners(self):
    """Return all four corners, starting from the origin, in CCW order."""
    return (
      (self.x, self.y),
      (self.x + self.width, self.y),
      (self.x + self.width, self.y + self.height),
      (self.x, self.y + self.height),
    )

  def get_num_onscreen_corners(self):
    """Return the number (max four) of corners that are on the image."""
    return sum(
      1 for x, y in self.get_corners()
      if (0 <= x < self.im_width) and (0 <= y < self.im_height))

  def quantize(self):
    """Creating a `BBox2D` with float values is technically OK; use this
    method to round to integer values in-place."""
    ATTRS = ('x', 'y', 'width', 'height', 'im_width', 'im_height')
    def quantize(v):
      return int(round(v)) if v is not None else v
    for attr in ATTRS:
      setattr(self, attr, quantize(getattr(self, attr)))

  def clamp_to_screen(self):
    """Clamp any out-of-image corners to edges of the image."""
    def clip_and_norm(v, max_v):
      return int(np.clip(v, 0, max_v).round())
    
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    x1 = clip_and_norm(x1, self.im_width - 1)
    y1 = clip_and_norm(y1, self.im_height - 1)
    x2 = clip_and_norm(x2, self.im_width - 1)
    y2 = clip_and_norm(y2, self.im_height - 1)
    self.set_x1_y1_x2_y2(x1, y1, x2, y2)
    
  def get_intersection_with(self, other):
    """Create a new `BBox2D` containing the intersection with `other`."""
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    ox1, oy1, ox2, oy2 = other.get_x1_y1_x2_y2()
    ix1 = max(x1, ox1)
    ix2 = min(x2, ox2)
    iy1 = max(y1, oy1)
    iy2 = min(y2, oy2)
    
    import copy
    intersection = copy.deepcopy(self)
    intersection.set_x1_y1_x2_y2(ix1, iy1, ix2, iy2)
    return intersection

  def get_union_with(self, other):
    """Create a new `BBox2D` containing the union with `other`."""
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    ox1, oy1, ox2, oy2 = other.get_x1_y1_x2_y2()
    ux1 = min(x1, ox1)
    ux2 = max(x2, ox2)
    uy1 = min(y1, oy1)
    uy2 = max(y2, oy2)
    
    import copy
    union = copy.deepcopy(self)
    union.set_x1_y1_x2_y2(ux1, uy1, ux2, uy2)
    return union

  def overlaps_with(self, other):
    """Does this `BBox2D` overlap with `other`."""
    # TODO: faster
    return self.get_intersection_with(other).get_area() > 0

  def get_area(self):
    """Area in square pixels"""
    return self.width * self.height

  def translate(self, *args):
    """Move the origin of this `BBox2D` by the given `(x, y)` value;
    either a tuple or a `numpy.ndarray`."""
    if len(args) == 1:
      x, y = args[0].tolist()
    else:
      x, y = args
    self.x += x
    self.y += y

  def get_crop(self, img):
    """Given the `numpy` array image `img`, return a crop based on this
    `BBox2D`."""
    c, r, w, h = self.x, self.y, self.width, self.height
    return img[r:r+h, c:c+w, :]

  def draw_in_image(
        self,
        img,
        identify_by='category_name',
        thickness=2):
    """Draw a bounding box in `np_image`.

    Args:
      img (numpy.ndarray): Draw in this image.
      identify_by (str): Auto-pick color and text by this attribute of
        this instance.
      thickness (int): Thickness of the line in pixels.
    """

    assert self.im_height == img.shape[0], (self.im_height, img.shape)
    assert self.im_width == img.shape[1], (self.im_width, img.shape)

    id_value = getattr(self, identify_by)

    from mft_utils.plotting import hash_to_rbg
    color = hash_to_rbg(id_value)

    from mft_utils.plotting import draw_bbox_in_image
    draw_bbox_in_image(
      img, self, color=color, thickness=thickness, label_txt=str(id_value))

