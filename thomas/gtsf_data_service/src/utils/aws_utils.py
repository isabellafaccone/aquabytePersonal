import glob
import os

import boto3


def get_matching_s3_keys(s3_client, bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


class SQSClient:
    """ interaction with SQS """

    def __init__(credentials):
        self.client = boto3.client('sqs',
                                   aws_access_key_id=credentials["aws_access_key_id"],
                                   aws_secret_access_key=credentials["aws_secret_access_key"],
                                   region_name="eu-west-1")
        self.queue_url = credentials["queue_url"]

    def receive_message():
        """ receive one message from sqs """
        response = self.client.receive_message(QueueUrl=self.queue_url)
        return response

    def delete_message():
        """ delete message """
        self.client.delete_message(QueueUrl=self.queue_url,
                                   ReceiptHandle=receipt_handle)


class S3Client:

    def __init__(credentials, folder):
        self.client = boto3.client('s3',
                                   aws_access_key_id=credentials["aws_access_key_id"],
                                   aws_secret_access_key=credentials["aws_secret_access_key"],
                                   region_name="eu-west-1")
        self.folder = folder


    def download(local_path, bucket, key):
        """download raw image from s3"""
        self.client.download_file(bucket, key, local_path)


    def download_raw_images(meta_bucket, meta_key):
        """ download raw images using metadata key """
        meta_folder = os.path.dirname(meta_key)
        raw_key = os.path.join(meta_folder, "raw")
        generator = get_matching_s3_keys(self.client,
                                         meta_bucket,
                                         prefix=raw_key,
                                         suffix=".jpg")
        for image_key in generator:
            image_name = os.path.basename(image_key)
            destination = os.path.join(self.folder, "raw", image_name)
            self.download(meta_bucket, image_key, destination)


    def upload_rectified(meta_bucket, meta_key):
        """upload rectified image to s3"""
        rectified_folder = os.path.join(self.folder, "rectified")
        rectified_images = glob.glob(rectified_folder + "/*.jpg")
        meta_folder = os.path.dirname(meta_key)
        for image_path in rectified_images:
            image_name = os.path.basename(image_path)
            rectified_key = os.path.join(meta_folder, "rectified", image_name)
            self.client.upload_file(image_path, meta_bucket, rectified_key)


    def parse_url(message):
        """parse the super clear sqs json"""
        bucket = eval(eval(message["Body"])['Message'])['Records'][0]["s3"]["bucket"]["name"]
        key = eval(eval(message["Body"])['Message'])['Records'][0]["s3"]['object']['key']
        return bucket, key
