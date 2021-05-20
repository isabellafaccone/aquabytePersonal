# MFT-PG Datasets

NB: You may want to symlink these to an external drive.
NB: You may want to [enable concurrent S3 requests](https://docs.aws.amazon.com/cli/latest/topic/s3-config.html)

To pull:
```
cd mft-pg/datasets
aws s3 sync --size-only s3://aquabyte-research/pwais/mft-pg/datasets_s3/ ./datasets_s3/
```

## Available Datasets:

### `datasets_s3/gopro1`

~1500 images pulled from a GoPro video with Fish and Fish Head
bounding boxes.  These images are split into `train` and `test`
chronologically-- the first half of the video is `train` and 
the second half is `test`.


### `datasets_s3/gopro_unlabeled1`

Over 70 minutes of unlabeled 30FPS GoPro footage (3840x2160).  Most of the
footage has the camera submerged with a production-like light pointed at the
fish.  Note that this dataset might have overlap with `datasets_s3/gopro1`
above. This dataset is from this directory on Aquabyte Google Drive:
https://drive.google.com/drive/u/0/folders/1hp-VDyQ2NpKKsG9TfLobjbwAOyDNeQJH


### `datasets_s3/high_recall_fish1`

These are 512px-by-512px crops from a production camera (at the time of
writing) that include fish bounding boxes (annotated as full or partial).

The dataset was pulled (and can be updated) as follows:

In a host shell (dockerized environment not required):
```
$ cd mft-pg/datasets/datasets_s3
$ mkdir -p high_recall_fish1/images && cd high_recall_fish1
$ aws s3 cp s3://aquabyte-images-adhoc/alok/training_datasets/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv ./
$ cd ../../..
$ python3
import pandas as pd
df = pd.read_csv('datasets/datasets_s3/high_recall_fish1/2021-05-20_high_recall_fish_detection_training_dataset_11011.csv')
uris = []
def parse(v):
  import ast
  lst = ast.literal_eval(v)
  assert len(lst) == 1
  return lst[0]

uris += [parse(r) for r in df['images']]
from mft_utils import misc as mft_misc
mft_misc.download_from_s3(uris, 'datasets/datasets_s3/high_recall_fish1/images')

```


### `datasets_s3/akpd1`

This dataset is a sample of AKPD data that Alok selected for 
https://aquabyte.atlassian.net/browse/ENGALL-2556 .  The dataset
has AKPD keypoints, left/right images, and camera intrinsics.


The dataset was pulled (and can be updated) as follows:

In a host shell (dockerized environment not required):
```
$ cd mft-pg/datasets/datasets_s3
$ mkdir -p akpd1/images && cd akpd1
$ aws s3 cp s3://aquabyte-images-adhoc/alok/training_datasets/2021-05-19_akpd_representative_training_dataset_10K.csv ./
$ cd ../../..
$ python3
import pandas as pd
df = pd.read_csv('datasets/datasets_s3/akpd1/2021-05-19_akpd_representative_training_dataset_10K.csv')
uris = []
uris += list(df['left_image_url'])
uris += list(df['right_image_url'])
def fixed(uri):
  if 'https://s3-eu-west-1.amazonaws.com/aquabyte-crops/' in uri:
    return uri.replace('https://s3-eu-west-1.amazonaws.com/aquabyte-crops/', 's3://aquabyte-crops/')
  elif 'https://aquabyte-crops.s3.eu-west-1.amazonaws.com/' in uri:
    return uri.replace('https://aquabyte-crops.s3.eu-west-1.amazonaws.com/', 's3://aquabyte-crops/')
  elif 'http://aquabyte-crops.s3.eu-west-1.amazonaws.com/' in uri:
    return uri.replace('http://aquabyte-crops.s3.eu-west-1.amazonaws.com/', 's3://aquabyte-crops/')
  elif 's3://aquabyte-crops/' in uri:
    return uri
  else:
    raise ValueError(uri)

uris = [fixed(u) for u in uris]
from mft_utils import misc as mft_misc
mft_misc.download_from_s3(uris, 'datasets/datasets_s3/akpd1/images')

```
