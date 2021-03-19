from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
import datetime
from typing import Dict, List
import urllib.parse
from datetime import date, datetime, time
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()

# from aq_util import str_util

# TODO: Old json missing annotationId won't work. Have to shore up the error reporting system and reject them outright.
# TODO: Now that crop_annotations is in its own module, move the static methods out to top level methods.


LEFT = 0
RIGHT = 1

# TODO: for backward compatibility, use this token if the input does not specify the model name
# TODO: rethink if there is a better way to communciate this information
DEFAULT_MODEL = 'ALL'
BATI_MODELS = [DEFAULT_MODEL, 'object-detection-bati']
LATI_MODELS = [DEFAULT_MODEL, 'object-detection']
BATI = "BATI"
LATI = "LATI"


@dataclass
class Crop:
    id           : str
    category     : Dict
    model        : Dict
    cropper_key  : str          # e.g. "environment=production/site-id=39/.../left_frame_crop_1516_1538_3436_3000.jpg"
    filename     : str
    src_frame    : str
    bbox         : List         # TODO: define what is inside
    bboxStandard : List         # TODO: define what is inside
    side         : int
    metadata     : Dict = field(default_factory=dict)
    pair_id      : str = None   # Unpaired BATI crop has pair_id of None
    url          : str = None   # public S3 URL of rectified image

    def for_service(self, service: str):
        name = self.model.get("name", DEFAULT_MODEL)
        if service == BATI:
            return name in BATI_MODELS
        elif service == LATI:
            return name in LATI_MODELS
        else:
            return False

    def __repr__(self):
        model = self.model["name"]
        cat_id = self.category["id"]
        return f"Crop(id={self.id}, side={self.side}, {model}, cat={cat_id}, {self.filename})"


def _escape(s):
    # `s` should be alphanumeric and underscore only
    # `_escape` is used only to prevent input not following the naming convention to ruin the data
    s = urllib.parse.quote(s)
    s = s.replace("-", "%2D")
    return s


def _filter_time(t):
    t = t.replace('000Z', '').replace('+00:00', '')
    # t = str_util.removesuffix(t, "000Z")
    # t = str_util.removesuffix(t, "+00:00")
    return t\
        .replace("-","")\
        .replace(":","")\
        .replace(" ","_")\
        .replace("T","_")

def format_count_attr(lst):
    counter = Counter(lst)
    return ', '.join(f'{k}: {v}' for k, v in counter.items())

class CropAnnotations(Mapping):

    # TODO: rethink what this is about. crops.json has information
    # scattered in "annotation", "cropper.crops", "category", "model".
    # This merge them into an useful object.
    # TODO: go further to design something useful. No need to
    # emulate the structure of crops.json.

    def __init__(self, crops_json=None, other=None):
        assert crops_json is not None or other is not None
        if crops_json is not None:
            self._load_crops_annotation(crops_json)
        else:
            self.crops_json    = other.crops_json
            self.is_full_frame = other.is_full_frame
            self.pen_id        = other.pen_id
            self.captured_at   = other.captured_at
            self.captured_at_dt= other.captured_at_dt
            self._items        = other._items
        self.crop_id = self.gen_crop_id(self.pen_id or "", self.captured_at or "")

    def _load_crops_annotation(self, crops_json):
        self.crops_json = crops_json

        # extract key attributes
        _capture = crops_json.get('capture') or {}
        _cropper = crops_json.get('cropper') or {}
        _crops = _cropper.get('crops') or []
        self.is_full_frame = _cropper.get('full_frame', False)
        self.pen_id = _capture.get('penId')
        self.captured_at = _capture.get('at')
        self.captured_at_dt = datetime.fromisoformat(self.captured_at[:26]) if self.captured_at else None

        # Generate an unique crop_id
        for i, crop in enumerate(_crops):
            crop['crop_id'] = str(i+1)

        _items = [self.make_crop(crop, crops_json) for crop in _crops]
        self._items = { a.id : a for a in _items }

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def select(self, service=None, side=None):
        other = CropAnnotations(other=self)
        if service:
            # pylint: disable=protected-access
            other._items = { c.id: c for c in self._items.values() if c.for_service(service)}
        if side is not None:
            # pylint: disable=protected-access
            other._items = { c.id: c for c in self._items.values() if c.side == side}
        return other

    @property
    def left(self):
        return [a for a in self._items.values() if a.side == LEFT]

    @property
    def right(self):
        return [a for a in self._items.values() if a.side == RIGHT]

    @classmethod
    def gen_crop_id(cls, pen_id: str, capturedAt: str):
        """ Build a short unique crop id """
        ps = _escape(pen_id)
        cs = _escape(_filter_time(capturedAt))
        return f"p{ps}-t{cs}"

    def get_pair_id(self, left_id=None, right_id=None):
        lc = f"{left_id}l" if left_id else ""
        rc = f"{right_id}r" if right_id else ""
        if lc and rc:
            return f"{self.crop_id}-c{lc}{rc}"
        else:
            return f"{self.crop_id}-c{lc or rc}"

    def report(self):
        return "{}\n{}".format(
            self,
            "\n".join(map(str, self._items.values()))
        )

    def __repr__(self):
        side_str = format_count_attr([c.side for c in self._items.values()])
        model_Str = format_count_attr([c.model['name'] for c in self._items.values()])
        return "CropAnnotations(id='{}', side={{{}}}, model={{{}}})".format(
            self.crop_id,
            side_str,
            model_Str,
        )

    @staticmethod
    def get_category(cat_id, crops_json):
        cat = next(x for x in crops_json['categories'] if x["id"] == cat_id)
        return cat

    @staticmethod
    def get_category_for_annotation(ann_id, crops_json):
        ann = next(i for i in crops_json['annotations'] if i["id"] == ann_id)
        cat_id = ann['category_id']
        cat_conf = ann['categoryConfidence']
        obj_conf = ann['objectConfidence']
        cat = CropAnnotations.get_category(cat_id, crops_json)
        single_cat = cat.copy()
        single_cat['categoryConfidence'] = cat_conf
        single_cat['objectConfidence'] = obj_conf
        return single_cat

    @staticmethod
    def get_model_for_annotation(category, crops_json):
        # TODO: use model dataclass rather than dict
        if 'model_id' not in category:
            # 2020-02
            # originally, there is one model 'object-detection' for all
            mi = crops_json.get('modelInfo', {})
            mi['name'] = DEFAULT_MODEL
            return mi

        m_id = category['model_id']
        m = next(x for x in crops_json['models'] if x["id"] == m_id)
        m.setdefault('name', DEFAULT_MODEL)
        return m

    def make_crop(self, cropper_crop, crops_json):
        # e.g. "environment=production/site-id=39/.../left_frame_crop_1516_1538_3436_3000.jpg"
        cropper_key = cropper_crop.get('key')
        if 'left' in cropper_key:
            side = LEFT
        elif 'right' in cropper_key:
            side = RIGHT
        else:
            assert False, cropper_key   # neither left or right

        filename = cropper_key if "/" not in cropper_key else cropper_key.rpartition("/")[2]

        crop_id = cropper_crop['crop_id']

        # (2020-08-29) src_frame is introduced to support offline full frame cropping (Mowi test)
        # e.g. "left_frame.jpg"
        src_frame = cropper_crop.get('src_frame')

        ann_id = cropper_crop.get('annotationID')
        categoryID = cropper_crop.get('categoryID')
        if ann_id:
            category = CropAnnotations.get_category_for_annotation(ann_id, crops_json)
        elif categoryID:
            category = CropAnnotations.get_category(categoryID, crops_json)
        else:
            # For backward compatibility if models missing then return 0
            # The code was written this way. Not sure if this is even a good idea.
            category = {"id": 0, "name": "salmon", "supercategory": "fish"}

        annotation = Crop(
            id           = crop_id,
            category     = category,
            model        = CropAnnotations.get_model_for_annotation(category, crops_json),
            cropper_key  = cropper_key,
            filename     = filename,
            src_frame    = src_frame,
            # TODO: this is BS. Use something like BBox
            bbox         = [int(cropper_crop['y0']), int(cropper_crop['x0']), int(cropper_crop['y1']), int(cropper_crop['x1'])],
            bboxStandard = [int(cropper_crop['x0']), int(cropper_crop['y0']), int(cropper_crop['x1']), int(cropper_crop['y1'])],
            side         = side,
        )
        return annotation