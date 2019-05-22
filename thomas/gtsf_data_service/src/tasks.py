import json
import os
import time

import boto3

from utils.aws_utils import S3Client, SqsClient
from utils.rectification_utils import Rectification
from utils.sql_utils import SqlClient


def main(base_folder):
    """
    - rectify the raw images
    - upload the rectified images to s3
    - detect the keypoints
    - update the database
    """
    aws_credentials = json.load(open(os.environ["AWS_CREDENTIALS"]))
    # create clients
    sqs = SqsClient(credentials)
    s3 = S3Client(credentials)
    sql = SqlClient(credentials)

    while True:
        response = sqs.receive_message()
        if "Messages" in response:
            # new messages in the queue
            message = response['Messages'][0]
            receipt_handle = message["ReceiptHandle"]

            # download the metadata json file
            meta_bucket, meta_key = s3.parse_url(message)
            meta_path = os.path.join(base_folder, "metadata.json")
            s3.download(meta_bucket, meta_key, meta_path)
            metadata = json.load(open(meta_path))
            # download the raw images
            s3.download_raw_images(meta_bucket, meta_key, base_folder)

            # get calibration and download the file locally
            calib_bucket, calib_key = sql.get_calibration(metadata["enclosure_id"])
            calibration_path = os.path.join(base_folder, "calibration.json")
            s3.download(calib_bucket, calib_key, calibration_path)

            # rectification time
            rectification = Rectification(calibration_path)

            # upload rectified images to s3
            s3.upload_rectfied(meta_bucket, meta_key)

            # add results to rds
            sql.populate_data_collection(meta_bucket, meta_key, metadata)

        time.sleep(10)


if __name__ == "__main__":
    main('/app/data/')
