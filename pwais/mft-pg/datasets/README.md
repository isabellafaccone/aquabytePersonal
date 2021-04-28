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


