import json
import os
import time

import boto3

from cogito_task import cogito_main
from gtsf_task import gtsf_main

credentials = json.load(open("credentials.json"))
s3_client = boto3.client('s3', aws_access_key_id=credentials["aws_access_key_id"],
                         aws_secret_access_key=credentials["aws_secret_access_key"],
                         region_name="eu-west-1")
new_size = (1024, 1024)


def main(base_folder, s3_client):
    while True:
        print('Looking at Cogito annotations....')
        cogito_main(os.path.join(base_folder, 'aquabyte-images'), s3_client, new_size)
        print('Looking at small pen images....')
        gtsf_main(os.path.join(base_folder, 'small_pen_data_collection'), s3_client)
        print('Now sleeping for 2 hours.....')
        time.sleep(7200)


if __name__ == "__main__":
    main('/root/data/', s3_client)
