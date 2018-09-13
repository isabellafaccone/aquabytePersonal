import json
import pandas as pd
import os
import keras
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.bin.train import makedirs
import math
from PIL import Image
import numpy as np
from skimage.measure import label
from scipy.misc import imsave

def csv_from_json(dataset_name, rez='low_rez', base_dir_dataset='/root/data/'):
    """
    Creates & saves a csv file suited for future detection training from json
    annotation file in COCO style dataset

    Input:
        - dataset_name: name of the dataset
        - rez: resolution of the blender images
        - base_dir_dataset: path of the directory which contains datasets

    Output:
        - csv files saved in dataset_path
    """
    dataset_path = base_dir_dataset + dataset_name
    target_dir = [dir for dir in os.listdir(
        dataset_path + '/training/') if rez in dir]
    for dir in target_dir:
        dir_path = dataset_path + '/training/' + dir + '/'
        dataset_list = []
        labels = json.load(open(dir_path + 'labels.json'))
        for label in labels:
            for bbox in label['bboxes']:
                dataset_list.append([label['path']] + bbox + ['fish'])
        pd.DataFrame(dataset_list).to_csv(dataset_path +
                                          '/annotations/' + dir + '_annotations.csv')

def get_bb_from_mask(mask, mask_value=255):
    """
    Computes the bounding box coordinates from the mask
    
    Input:
        - mask : np.array of size (L, H, 3)
        - mask_value: float, 
    
    Output:
        - (x1, y1, x2, y2) : coordinates of the corner of
        the bounding box
    """
    x_end, x_start = (np.where(mask[..., 0] == mask_value)[0].max(),
                      np.where(mask[..., 0] == mask_value)[0].min())
    y_end, y_start = (np.where(mask[..., 0] == mask_value)[1].max(), 
                      np.where(mask[..., 0] == mask_value)[1].min())
    
    return (y_start, x_start, y_end, x_end)  

def get_bboxes(annotations_path):
    """
    Generates a list of bboxes from mask instances path
    
    Input:
    - annotations_path: list of mask path
    
    Output:
    - bboxes: list of (x1, y1, x2, y2)
    """
    bboxes = []
    for path in annotations_path:
        mask = np.array(Image.open(path))[..., :3]
        if len(np.unique(mask)) > 1:
            bboxes.append(get_bb_from_mask(mask))
        else:
            bboxes.append((None, None, None, None))
    return bboxes

def create_instance_csv(dataset_path, annotations_dir, frames_dir,
                        target_path='/root/data/gopro/annotations/'):
    """
    Creates a csv file suited for the CSV custom dataset and saves
    it in the dataset_path
    Each raw has the following format:
    img_path, x1, y1, x2, y2, mask_path

    Input:
        - dataset_path: str, path of the dataset
        - annotations_dir: str, path of the annotations
    """
    if dataset_path[-1] != '/':
        dataset_path += '/'
    if target_path[-1] != '/':
        target_path += '/'
    annotations_path = sorted([dataset_path + annotations_dir + '/' + file for file in
                        os.listdir(dataset_path + annotations_dir)
                        if len(file.split('.', 2)[-1]) == 5 and file.split('.')[-2] in ['1']])
    class_ID = [mask.split('.', 3)[-2] for mask in annotations_path]
    images_path = [dataset_path + frames_dir + '/' + 
                   x.split('/', 5)[-1].split('.', 3)[0] + '.png' 
                   for x in annotations_path]
    bboxes = get_bboxes(annotations_path)
    x_0 = [x[0] for x in bboxes]
    y_0 = [x[1] for x in bboxes]
    x_1 = [x[2] for x in bboxes]
    y_1 = [x[3] for x in bboxes]
    dataframe = pd.DataFrame([images_path, x_0, y_0, x_1, y_1, class_ID]).T
    dataframe = dataframe.dropna()
    dataframe.to_csv(target_path + 'annotations.csv', header=None, index=False)
    print('csv file created at ' + target_path)
    
    return dataframe