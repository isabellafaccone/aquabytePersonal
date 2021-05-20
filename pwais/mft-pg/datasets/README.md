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


### `datasets_s3/akpd1`

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