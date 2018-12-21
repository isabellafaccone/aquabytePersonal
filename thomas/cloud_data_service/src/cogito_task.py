import copy
import json
import os
import urllib
from datetime import datetime

import skimage.io as io
from skimage.transform import resize
from pycocotools.coco import COCO

from utils import get_matching_s3_keys


def cogito_main(base_folder, s3_client, new_size):
    """ every hour check s3 folder for new files"""
    generator = get_matching_s3_keys(s3_client,
                                     'aquabyte-annotations',
                                     prefix='cogito/to-be-processed',
                                     suffix='.json')

    for key in generator:
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
        annotations_resized = copy.deepcopy(annotations)

        # step 0 - take care of annotations
        # download the images into the corresponding folders
        for (i, (annotation, annotation_res)) in enumerate(zip(annotations['images'], annotations_resized['images'])):
            if i % 1000 == 0:
                print('Image {} out of {} downloaded and added'.format(i, len(annotations['images'])))
            url = annotation['coco_url']
            assert annotation['coco_url'] == annotation_res['coco_url'], "Problem!!"

            image_name = url.split('%2F')[-1].split('?')[0]
            farm = image_name.split('_')[1]
            pen = image_name.split('_')[2]
            date = str(datetime.utcfromtimestamp(int(image_name.split('_')[-1].split('.')[0])/1000.0).date())
            image_dir = os.path.join(base_folder, farm, date, pen)
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            image_destination = os.path.join(image_dir, image_name)
            if not os.path.isfile(image_destination):
                urllib.urlretrieve(url, image_destination)

            image_resized_destination = image_destination.replace("aquabyte-images", "aquabyte-images-resized")

            if not os.path.isdir(os.path.dirname(image_resized_destination)):
                os.makedirs(os.path.dirname(image_resized_destination))
            if not os.path.isfile(image_resized_destination):
                image = io.imread(image_destination)
                image_resized = resize(image, new_size)
                io.imsave(image_resized_destination, image_resized)

            annotation["local_path"] = image_destination
            annotation_res['height'] = new_size[0]
            annotation_res['width'] = new_size[0]
            annotation_res["local_path"] = image_resized_destination

        with open(os.path.join(base_folder, 'cocofiles', 'coco_body_parts_' + json_file), 'w') as f:
            json.dump(annotations, f)

        # step 2 - save heads with eye
        id2class = json.load(open("./id_class_matching.json"))
        coco = COCO(os.path.join(base_folder, 'cocofiles', 'coco_body_parts_' + json_file))


        # step 3 - take care of resized annotations
        yfactor = new_size[0] / 3000.0
        xfactor = new_size[1] / 4096.0
        # resize the annotations as well
        for (j, ann) in enumerate(annotations_resized['annotations']):
            if j % 50 == 0:
                print('Annotation {} out of {} resized'.format(j, len(annotations_resized['annotations'])))
            # bbox
            bbox = ann['bbox']
            bbox_resized = [int(bbox[0]*xfactor), int(bbox[1]*yfactor), int(bbox[2]*xfactor), int(bbox[3]*yfactor)]
            ann['bbox'] = bbox_resized

            # segmentation
            seg = ann['segmentation'][0]
            seg_resized = []
            for (i, v) in enumerate(seg):
                if i % 2 == 0:
                    factor = xfactor
                else:
                    factor = yfactor
                seg_resized.append(int(v*factor))
            ann['segmentation'] = [seg_resized]

        with open(os.path.join(base_folder.replace('aquabyte-images', 'aquabyte-images-resized'), 'cocofiles',
                               'coco_body_parts_' + json_file), 'w') as f:
            json.dump(annotations_resized, f)



        # @TODO Old code
        # # create a coco for each type
        # coco_visibility = {
        #     "info": INFO,
        #     "licenses": LICENSES,
        #     "categories": CATEGORIES,
        #     "images": [],
        #     "annotations": []
        # }
        #
        # coco_orientation = {
        #     "info": INFO,
        #     "licenses": LICENSES,
        #     "categories": CATEGORIES,
        #     "images": [],
        #     "annotations": []
        # }
        #
        # coco_occlusion = {
        #     "info": INFO,
        #     "licenses": LICENSES,
        #     "categories": CATEGORIES,
        #     "images": [],
        #     "annotations": []
        # }
        # image_id = 1
        # segmentation_id = 1
        #
        # json_file = os.path.basename(key)
        # json_destination = os.path.join(base_folder, 'processed', json_file)
        #
        # # check if the file has been downloaded
        # if os.path.isfile(json_destination):
        #     continue
        #
        # # otherwise download the file
        # print('A new json file has been found {}. Downloading it!!'.format(key))
        # s3_client.download_file("aquabyte-annotations", key, json_destination)
        #
        # # open the downloaded file
        # annotations = json.load(open(json_destination))
        #
        # # download the images into the corresponding folders
        # for (i, annotation) in enumerate(annotations['Images']):
        #     if i % 50 == 0:
        #         print('Image {} out of {} downloaded and added'.format(i, len(annotations['Images'])))
        #     url = annotation['ImageURL']
        #     image_name = os.path.basename(url)
        #     farm = annotation['FarmName']
        #     pen = annotation['PenId']
        #     date = annotation['CapturedDate']
        #     image_dir = os.path.join(base_folder, farm, date, pen)
        #     if not os.path.isdir(image_dir):
        #         os.makedirs(image_dir)
        #     image_destination = os.path.join(image_dir, image_name)
        #     if not os.path.isfile(image_destination):
        #         urllib.urlretrieve(url, image_destination)
        #
        #     # open the image and add the image info
        #     image = Image.open(image_destination)
        #     image_info = create_image_info(image_id, image_destination, image.size)
        #     coco_visibility["images"].append(image_info)
        #     coco_occlusion["images"].append(image_info)
        #     coco_orientation["images"].append(image_info)
        #
        #     # go through the polygons
        #     # labels = ast.literal_eval(annotation['Annotations'])
        #     labels = annotation['Annotations']
        #     for label in labels:
        #         polygon = [(item['X'], item['Y']) for item in label['Coordinates']]
        #         img = Image.new('L', (image.width, image.height), 0)
        #         ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        #
        #         binary_mask = np.array(img).astype(np.uint8)
        #
        #         # visibility
        #         category_info = {'id': visibility[label['VisibilityType']],
        #                          'is_crowd': 'crowd' in image_destination}
        #         annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask,
        #                                                  image.size, tolerance=2)
        #         if annotation_info is not None:
        #             coco_visibility["annotations"].append(copy.copy(annotation_info))
        #
        #         # occlusion
        #         annotation_info['category_id'] = occlusion[label['OcclusionType']]
        #         if annotation_info is not None:
        #             coco_occlusion["annotations"].append(copy.copy(annotation_info))
        #
        #         # orientation
        #         annotation_info['category_id'] = orientation[label['OrientationType']]
        #         if annotation_info is not None:
        #             coco_orientation["annotations"].append(copy.copy(annotation_info))
        #
        #         segmentation_id += 1
        #     image_id += 1
        #
        # with open(os.path.join(base_folder, 'cocofiles', 'coco_visibility_' + json_file), 'w') as f:
        #     json.dump(coco_visibility, f)
        # with open(os.path.join(base_folder, 'cocofiles', 'coco_orientation_' + json_file), 'w') as f:
        #     json.dump(coco_orientation, f)
        # with open(os.path.join(base_folder, 'cocofiles', 'coco_occlusion_' + json_file), 'w') as f:
        #     json.dump(coco_occlusion, f)
        #
        # # post processing time - this takes a fair amount of time
        # print('Post processing starting.....')
        # create_combinations_files(json_destination, INFO, LICENSES, CATEGORIES)

