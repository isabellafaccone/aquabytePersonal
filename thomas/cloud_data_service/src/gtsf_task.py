import glob
import os

from utils import get_matching_s3_keys, Rectification


def gtsf_main(base_folder, s3_client):
    result = s3_client.list_objects(Bucket='aquabyte-groundtruths', Prefix='rnd/small-pen-test-site/1/1', Delimiter='/')
    cloud_experiments = [exp['Prefix'].split('/')[-2] for exp in result['CommonPrefixes']]

    local_experiments = os.listdir(base_folder)
    cloud_experiments_download = [ce for ce in cloud_experiments if ce not in local_experiments]

    # first list the keys and download the data
    print('Now downloading data from {} experiments'.format(len(cloud_experiments_download)))
    for exp in cloud_experiments_download:
        # create the local folder
        os.makedirs(os.path.join(base_folder, exp))
        generator = get_matching_s3_keys(s3_client,
                                         'aquabyte-groundtruths',
                                         prefix='rnd/small-pen-test-site/1/{}/'.format(exp),
                                         suffix='')
        print('Downloading images for experience {}'.format(exp))
        for key in generator:
            destination = os.path.join(base_folder, exp, os.path.basename(key))
            s3_client.download_file("aquabyte-groundtruths", key, destination)

    # second, rectification
    rectification_files = glob.glob(os.path.join(base_folder,
                                                 '/underwater_enclosure_0*/parameters*'))
    Rectification(base_folder, cloud_experiments_download, rectification_files)
