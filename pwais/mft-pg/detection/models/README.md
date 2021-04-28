# MFT-PG Detection Models

NB: You may want to [enable concurrent S3 requests](https://docs.aws.amazon.com/cli/latest/topic/s3-config.html)

To pull:
```
cd mft-pg/detection/models
aws s3 sync --size-only s3://aquabyte-research/pwais/mft-pg/detection_models_s3/ ./detection_models_s3/
```

## Available Models:

### `detection_models_s3/yolo_ragnarok_config_hack0`

Trained on the union of the train and test splits of `datasets_s3/gopro1`
(the model predates those splits), this Yolov3 model uses a config stolen
from ragnarok (that should happen to be close to what is used in the cloud in 
production at the time of writing):
https://github.com/aquabyte-new/ragnarok/blob/2294ffc1734bb8df1c0227a28e982afc4b52b1f7/deepservice/modules/cropper/yolov3-fish.cfg#L2

This detector was used for an initial Yolo+SORT tracking demo:
https://drive.google.com/file/d/18v0LP9JRSqeEog6wrrFugeM6n-3IehZZ/view?usp=sharing

