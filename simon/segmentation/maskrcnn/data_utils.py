import datetime
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
from cococreatortools import create_image_info, filter_for_annotations, create_annotation_info
import json
import random
from shutil import copyfile

def generate_annotations_json(dataset_path, mask_dir, frame_dir,
                              dataset_name, class_ID_path, 
                              multi_class=False, split=0.8):
    """
    Generates the training & test json coco dataset file

    Input:
         - dataset_path: str, path of the dataset
         - mask_dir: str, name of the mask directory
         - dataset_name: str, name of the dataset
         - class_ID: str, path of the class ID file
    """
    INFO = {"description": dataset_name,
            "date_created": datetime.datetime.utcnow().isoformat(' ')}
    CATEGORIES = []
    class_ID = pd.read_csv(class_ID_path, header=None)
    if multi_class:
        for i in range(len(class_ID)):
            CATEGORIES.append({'id:': class_ID[i][1],
                               'name': class_ID[i][0]})
    else:
        CATEGORIES.append({'id': 1,
                           'name': 'fish',})

    image_files = list(set([dataset_path + frame_dir + '/' + x.split('_', 1)[0]
                            for x in os.listdir(dataset_path + mask_dir)]))
    random.shuffle(image_files)
    train_image_files = image_files[:int(split * len(image_files))]
    test_image_files = image_files[int(split * len(image_files)):]
    coco_output_train = {'categories': CATEGORIES,
                   'info': INFO,
                   'images': [],
                   'annotations': []}
    coco_output_test = {'categories': CATEGORIES,
                   'info': INFO,
                   'images': [],
                   'annotations': []}
    fill_annotation_dict(coco_output_train, train_image_files, dataset_path, 
                         mask_dir, multi_class)
    fill_annotation_dict(coco_output_test, test_image_files, dataset_path, 
                         mask_dir, multi_class)
    train_dataset_name = dataset_name + '_train'
    test_dataset_name = dataset_name + '_test'
    create_json(dataset_path, train_dataset_name, coco_output_train)
    create_json(dataset_path, test_dataset_name, coco_output_test)

    return coco_output_train, coco_output_test

def fill_annotation_dict(coco_output, image_files, dataset_path, mask_dir, 
                         multi_class):
    """
    Fill the coco json file

    Input:
        - coco_output: dict, coco json file
        - image_files: list, list of image files
        - dataset_path: str, path of the dataset
        - mask_dir: str, name of the mask directory

    Output:
        - coco_output: dict
    """
    image_id = 1
    segmentation_id = 1

    for ix in trange(len(image_files)):
        image = Image.open(image_files[ix])
        image_info = create_image_info(image_id,
                                       os.path.basename(image_files[ix]),
                                       image.size)
        coco_output['images'].append(image_info)

        for root, _, files in os.walk(dataset_path + mask_dir):
            annotations_files = filter_for_annotations(root, 
                                                       files,
                                                       image_files[ix])
            
            for annotation_filename in annotations_files:
                if multi_class:
                    class_id = int(annotation_filename.\
                               split('.', annotation_filename.count('.'))[-2])
                else:
                    class_id = 1
                category_info = {'id': class_id, 'is_crowd': 'crowd' in
                                 image_files[ix]}
                binary_mask = np.asarray(Image.open(annotation_filename)
                              .convert('1')).astype(np.uint8)
                annotation_info = create_annotation_info(segmentation_id, 
                                                         image_id, 
                                                         category_info, 
                                                         binary_mask,
                                                         image.size, 
                                                         tolerance=2)
                if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                segmentation_id += 1

        image_id +=1

def create_json(dataset_path, dataset_name, coco_output):
    """
    Creates a json file from dict

    Input:
         - dataset_path: str, path of the dataset
         - dataset_name: str, name of the dataset
         - dict
    """
    with open('{}/instance_{}.json'.\
    format(dataset_path, dataset_name),'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    print('File annotation_{} created at {}'.format(dataset_name,
                                                    dataset_path))

def create_training_folder_structure(train_json, test_json, train_target_dir, 
									 test_target_dir, dataset_path, frame_dir):
	"""
	Creates the folder structure suited for the COCO Dataset generator

	Input:
	     - train_json: str, name of the coco json file for train images
	     - test_json_path: str, name of the coco json file for test images
	     - train_target_dir: str, name of the train target dir
	     - test_target_dir: str, name of the test target dir
	     - dataset_path: str, path of the dataset
         - frame_idr: str, name of the frame dir
	"""
	if dataset_path[-1]!= '/':
		dataset_path += '/'
	train_json_path = dataset_path + 'instance_annotations/' + train_json
	test_json_path = dataset_path +  'instance_annotations/' + test_json
	with open(train_json_path) as f:
		data_train = json.load(f)
	with open(test_json_path) as f:
		data_test = json.load(f)
	train_images = [data_train['images'][i]['file_name'] for i in 
					range(len(data_train['images']))]
	test_images = [data_test['images'][i]['file_name'] for i in 
				   range(len(data_test['images']))]
	train_target_path = dataset_path + train_target_dir
	test_target_path = dataset_path + test_target_dir

	# Copy train images
	for i in trange(len(train_images)):
		copyfile(dataset_path + frame_dir + '/' + train_images[i],
				 train_target_path + '/' + train_images[i])
	print('Images copied to ' + train_target_path)

	# Copy test images
	for i in trange(len(test_images)):
		copyfile(dataset_path + frame_dir + '/' + test_images[i],
				 test_target_path + '/' + test_images[i])
	print('Images copied to ' + test_target_dir)