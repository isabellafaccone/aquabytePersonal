import boto3
import tempfile

def get_bucket_key(url):
    # old style https://s3.amazonaws.com/<bucket>/<key>
    # new style https://<bucket>.s3.amazonaws.com/<key>
    splitted = url.split("/")
    # eg s3.amazonaws.com
    first_part = splitted[2].split('.')
    if len(first_part) != 3:
        # new style
        bucket = first_part[0]
        key = "/".join(splitted[3:])
    else:
        bucket = splitted[3]
        key = "/".join(splitted[4:])
    return bucket, key

def get_stream(bucket, key):
    s3 = boto3.resource('s3')
    tmp = tempfile.NamedTemporaryFile(delete=False)
    bucket = s3.Bucket(bucket)
    obj = bucket.Object(key)
    return obj
