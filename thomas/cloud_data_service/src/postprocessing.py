import copy
import json
import os
import urllib

import numpy as np
from PIL import Image, ImageDraw

from pycococreator import create_image_info, create_annotation_info


def create_combinations_files(json_path, info, licenses, categories, base_folder='/root/data/aquabyte-images/'):
    """ from the json downloaded from cogito, created multiple sub json files, one per combination"""
    coco_clear_full_curved = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }
    coco_clear_full_lateral = copy.deepcopy(coco_clear_full_curved)
    coco_clear_full_other = copy.deepcopy(coco_clear_full_curved)
    coco_clear_partial_curved = copy.deepcopy(coco_clear_full_curved)
    coco_clear_partial_lateral = copy.deepcopy(coco_clear_full_curved)
    coco_clear_partial_other = copy.deepcopy(coco_clear_full_curved)

    image_counter = {'clear_full_curved': 1,
                     'clear_full_lateral': 1,
                     'clear_full_other': 1,
                     'clear_partial_curved': 1,
                     'clear_partial_lateral': 1,
                     'clear_partial_other': 1, }
    segmentation_counter = {'clear_full_curved': 1,
                            'clear_full_lateral': 1,
                            'clear_full_other': 1,
                            'clear_partial_curved': 1,
                            'clear_partial_lateral': 1,
                            'clear_partial_other': 1, }

    json_file = os.path.basename(json_path)
    json_destination = os.path.join(base_folder, 'processed', json_file)

    # open the downloaded file
    annotations = json.load(open(json_destination))

    for (i, annotation) in enumerate(annotations['Images']):
        if i % 50 == 0:
            print ('Image {} out of {} postprocessed'.format(i, len(annotations['Images'])))

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
        image_id = 0  # dummy variable
        image_info = create_image_info(image_id, image_destination, image.size)

        # go through the polygons
        # labels = ast.literal_eval(annotation['Annotations'])
        labels = annotation['Annotations']

        is_image = {'clear_full_curved': 0,
                    'clear_full_lateral': 0,
                    'clear_full_other': 0,
                    'clear_partial_curved': 0,
                    'clear_partial_lateral': 0,
                    'clear_partial_other': 0, }

        for label in labels:
            try:
                polygon = [(item['X'], item['Y']) for item in label['Coordinates']]
                img = Image.new('L', (image.width, image.height), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

                binary_mask = np.array(img).astype(np.uint8)

                viz = label['VisibilityType']
                occ = label['OcclusionType']
                ori = label['OrientationType']
                category_info = {'id': 0, 'is_crowd': 'crowd' in image_destination}

                segmentation_id = 0  # dummy variable
                annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask,
                                                         image.size, tolerance=2)

                if annotation_info is not None:
                    if viz == 'clear' and occ == 'full' and ori == 'lateral':
                        if is_image['clear_full_lateral'] == 0:
                            # add image
                            is_image['clear_full_lateral'] += 1
                            image_info['id'] = image_counter['clear_full_lateral']
                            image_counter['clear_full_lateral'] += 1
                            coco_clear_full_lateral['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_full_lateral']
                        annotation_info['image_id'] = image_counter['clear_full_lateral'] - 1
                        segmentation_counter['clear_full_lateral'] += 1
                        coco_clear_full_lateral["annotations"].append(copy.copy(annotation_info))

                    elif viz == 'clear' and occ == 'full' and ori == 'curved':
                        if is_image['clear_full_curved'] == 0:
                            # add image
                            is_image['clear_full_curved'] += 1
                            image_info['id'] = image_counter['clear_full_curved']
                            image_counter['clear_full_curved'] += 1
                            coco_clear_full_curved['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_full_curved']
                        annotation_info['image_id'] = image_counter['clear_full_curved'] - 1
                        segmentation_counter['clear_full_curved'] += 1
                        coco_clear_full_curved["annotations"].append(copy.copy(annotation_info))

                    elif viz == 'clear' and occ == 'full' and ori == 'other':
                        if is_image['clear_full_other'] == 0:
                            # add image
                            is_image['clear_full_other'] += 1
                            image_info['id'] = image_counter['clear_full_other']
                            image_counter['clear_full_other'] += 1
                            coco_clear_full_other['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_full_other']
                        annotation_info['image_id'] = image_counter['clear_full_other'] - 1
                        segmentation_counter['clear_full_other'] += 1
                        coco_clear_full_other["annotations"].append(copy.copy(annotation_info))

                    elif viz == 'clear' and occ == 'partial' and ori == 'lateral':
                        if is_image['clear_partial_lateral'] == 0:
                            # add image
                            is_image['clear_partial_lateral'] += 1
                            image_info['id'] = image_counter['clear_partial_lateral']
                            image_counter['clear_partial_lateral'] += 1
                            coco_clear_partial_lateral['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_partial_lateral']
                        annotation_info['image_id'] = image_counter['clear_partial_lateral'] - 1
                        segmentation_counter['clear_partial_lateral'] += 1
                        coco_clear_partial_lateral["annotations"].append(copy.copy(annotation_info))

                    elif viz == 'clear' and occ == 'partial' and ori == 'curved':
                        if is_image['clear_partial_curved'] == 0:
                            # add image
                            is_image['clear_partial_curved'] += 1
                            image_info['id'] = image_counter['clear_partial_curved']
                            image_counter['clear_partial_curved'] += 1
                            coco_clear_partial_curved['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_partial_curved']
                        annotation_info['image_id'] = image_counter['clear_partial_curved'] - 1
                        segmentation_counter['clear_partial_curved'] += 1
                        coco_clear_partial_curved["annotations"].append(copy.copy(annotation_info))


                    elif viz == 'clear' and occ == 'partial' and ori == 'other':
                        if is_image['clear_partial_other'] == 0:
                            # add image
                            is_image['clear_partial_other'] += 1
                            image_info['id'] = image_counter['clear_partial_other']
                            image_counter['clear_partial_other'] += 1
                            coco_clear_partial_other['images'].append(copy.copy(image_info))
                        # add annotations
                        annotation_info['id'] = segmentation_counter['clear_partial_other']
                        annotation_info['image_id'] = image_counter['clear_partial_other'] - 1
                        segmentation_counter['clear_partial_other'] += 1
                        coco_clear_partial_other["annotations"].append(copy.copy(annotation_info))

            except Exception as e:
                print e.message

            segmentation_id += 1
        image_id += 1
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_full_lateral_' + json_file), 'w') as f:
        json.dump(coco_clear_full_lateral, f)
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_full_curved_' + json_file), 'w') as f:
        json.dump(coco_clear_full_curved, f)
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_full_other_' + json_file), 'w') as f:
        json.dump(coco_clear_full_other, f)
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_partial_lateral_' + json_file), 'w') as f:
        json.dump(coco_clear_partial_lateral, f)
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_partial_curved_' + json_file), 'w') as f:
        json.dump(coco_clear_partial_curved, f)
    with open(os.path.join(base_folder, 'cocofiles', 'coco_clear_partial_other_' + json_file), 'w') as f:
        json.dump(coco_clear_partial_other, f)


