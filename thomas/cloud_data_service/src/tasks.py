import ast
import copy
import json
import os
import time
import urllib
from datetime import datetime
from PIL import Image, ImageDraw

import boto3
import numpy as np

from postprocessing import create_combinations_files
from pycococreator import create_image_info, create_annotation_info
from utils import get_matching_s3_keys


INFO = {
    "description": "Fish data",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "thossler",
    "date_created": datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'salmon',
        'supercategory': 'fish',
    },
    {
        'id': 1,
        'name': 'clear',
        'supercategory': 'fish',
    },
    {
        'id': 2,
        'name': 'full',
        'supercategory': 'fish',
    },
    {
        'id': 3,
        'name': 'partial',
        'supercategory': 'fish',
    },
    {
        'id': 4,
        'name': 'curved',
        'supercategory': 'fish',
    },
    {
        'id': 5,
        'name': 'lateral',
        'supercategory': 'fish',
    },
    {
        'id': 6,
        'name': 'other',
        'supercategory': 'fish',
    }
]

# matching keys to ids
visibility = {'clear': 1}
occlusion = {'full': 2, 'partial': 3}
orientation = {'curved': 4, 'lateral': 5, 'other': 6}


credentials = json.load(open("credentials.json"))
s3_client = boto3.client('s3', aws_access_key_id=credentials["aws_access_key_id"],
                         aws_secret_access_key=credentials["aws_secret_access_key"],
                         region_name="eu-west-1")


def main(base_folder):
    """ every hour check s3 folder for new files"""
    while True:
        generator = get_matching_s3_keys(s3_client,
                                         'aquabyte-annotations',
                                         prefix='cogito/to-be-processed',
                                         suffix='.json')

        for key in generator:
            # create a coco for each type
            coco_visibility = {
                "info": INFO,
                "licenses": LICENSES,
                "categories": CATEGORIES,
                "images": [],
                "annotations": []
            }

            coco_orientation = {
                "info": INFO,
                "licenses": LICENSES,
                "categories": CATEGORIES,
                "images": [],
                "annotations": []
            }

            coco_occlusion = {
                "info": INFO,
                "licenses": LICENSES,
                "categories": CATEGORIES,
                "images": [],
                "annotations": []
            }
            image_id = 1
            segmentation_id = 1

            json_file = os.path.basename(key)
            json_destination = os.path.join(base_folder, 'processed', json_file)

            # check if the file has been downloaded
            if os.path.isfile(json_destination):
                continue

            # otherwise download the file
            print('A new json file has been found {}. Downloading it!!'.format(key))
            s3_client.download_file("aquabyte-annotations", key, json_destination)

            # open the downloaded file
            annotations = json.load(open(json_destination))

            # download the images into the corresponding folders
            for (i, annotation) in enumerate(annotations['Images']):
                if i % 50 == 0:
                    print('Image {} out of {} downloaded and added'.format(i, len(annotations['Images'])))
                url = annotation['ImageURL']
                image_name = os.path.basename(url)
                farm = annotation['FarmName']
                pen = annotation['PenId']
                date = annotation['CapturedDate']
                image_dir = os.path.join(base_folder, farm, date, pen)
                if not os.path.isdir(image_dir):
                    os.makedirs(image_dir)
                image_destination = os.path.join(image_dir, image_name)
                if not os.path.isfile(image_destination):
                    urllib.urlretrieve(url, image_destination)

                # open the image and add the image info
                image = Image.open(image_destination)
                image_info = create_image_info(image_id, image_destination, image.size)
                coco_visibility["images"].append(image_info)
                coco_occlusion["images"].append(image_info)
                coco_orientation["images"].append(image_info)

                # go through the polygons
                # labels = ast.literal_eval(annotation['Annotations'])
                labels = annotation['Annotations']
                for label in labels:
                    polygon = [(item['X'], item['Y']) for item in label['Coordinates']]
                    img = Image.new('L', (image.width, image.height), 0)
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

                    binary_mask = np.array(img).astype(np.uint8)

                    # visibility
                    category_info = {'id': visibility[label['VisibilityType']],
                                     'is_crowd': 'crowd' in image_destination}
                    annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask,
                                                             image.size, tolerance=2)
                    if annotation_info is not None:
                        coco_visibility["annotations"].append(copy.copy(annotation_info))

                    # occlusion
                    annotation_info['category_id'] = occlusion[label['OcclusionType']]
                    if annotation_info is not None:
                        coco_occlusion["annotations"].append(copy.copy(annotation_info))

                    # orientation
                    annotation_info['category_id'] = orientation[label['OrientationType']]
                    if annotation_info is not None:
                        coco_orientation["annotations"].append(copy.copy(annotation_info))

                    segmentation_id += 1
                image_id += 1

            with open(os.path.join(base_folder, 'cocofiles', 'coco_visibility_' + json_file), 'w') as f:
                json.dump(coco_visibility, f)
            with open(os.path.join(base_folder, 'cocofiles', 'coco_orientation_' + json_file), 'w') as f:
                json.dump(coco_orientation, f)
            with open(os.path.join(base_folder, 'cocofiles', 'coco_occlusion_' + json_file), 'w') as f:
                json.dump(coco_occlusion, f)

            # post processing time - this takes a fair amount of time
            print('Post processing.....')
            create_combinations_files(json_destination, INFO, LICENSES, CATEGORIES)

        print('sleeping for two hours now....')
        time.sleep(7200)


if __name__ == "__main__":
    main('/root/data/aquabyte-images/')
