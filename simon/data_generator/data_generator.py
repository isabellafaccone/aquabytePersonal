import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label
from scipy.misc import imsave
import time
from tqdm import tqdm

def _reconstruct_semantic_mask(mask,
                               contour_color,
                               class_color_dict={1: [255, 105, 180],
                                                0: [0, 0, 255]}, 
                               version='gopro'):
    """
    Returns a reconstructed mask with dilated contours & eroded masks for
    each class.
    All instances of a given class have a color defined by the class_color_dict
    The color of the contour is defined by contour_color list

    Input:
        - mask: np.array of size (W, H, 3)
        - class_color_dict: dict of color class class_id {class_id: [R, G, B]}
        - contour_color: list of [R, G, B]

    Output:
        - reconstructed_mask: np.array of size (W, H, 3)
    """
    red, green, blue = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    class_masks = []
    #
    for i in class_color_dict.keys():
        class_masks.append((i,
                            (red == class_color_dict[i][0]) &
                            (green == class_color_dict[i][1]) &
                            (blue == class_color_dict[i][2])
                            ))
    contours = np.logical_and(red > 0, red < 254) &\
        np.logical_and(green > 0, green < 254) &\
        np.logical_and(blue > 0, blue <= 255)
    dilated_contours = cv2.dilate((contours * 1.0).astype(np.float32),
                                  np.ones((10, 10)))
    eroded_class_masks = [(x[0], cv2.dilate((x[1] * 1.0).
                                            astype(np.float32),
                                            np.ones((10, 10))))
                          for x in class_masks]
    reconstructed_mask = np.zeros(mask.shape)
    for m in class_masks:
        mask_indices = np.where(m[1] == 1)
        reconstructed_mask[mask_indices[0],
                           mask_indices[1], :] = class_color_dict[m[0]]
    contours_indices = np.where(dilated_contours == 1)
    reconstructed_mask[contours_indices] = contour_color

    return reconstructed_mask.reshape(mask.shape)

def generate_semantic_masks(dataset_path, semantic_dir, target_dir,
                            contour_color,
                            class_color_dict={1: [255, 105, 180],
                                              0: [0, 0, 255]},
                            version='gopro'):
    """
    Generates semantic masks dilated contours & eroded masks

    Input: 
        - dataset_path: str, path of the dataset
        - semantic_mask_dir: str, name of the directory which contains semantic
        masks
        - target_dir: str, name of the directory in which we save semantic 
        masks
        - class_color_dict : dict of color & class_id {class_id: [R, G, B]}
        - contour_color : list of contour color [R, G, B]
    """
    if version == 'gopro':
        img_names = [file.split('.', 1)[0] for file in
                     os.listdir(dataset_path + semantic_dir) if
                     file.split('.', 1)[-1] == 'semantic.png']
        masks_path = [dataset_path + semantic_dir + '/' + file + 
                      '.semantic.png' for file in img_names]
    else:
        img_names = [file.split('.', 1)[0] for file in 
                     os.listdir(dataset_path + semantic_dir)]
        masks_path = [dataset_path + semantic_dir + '/' + file +
                      '.png' for file in img_names]
    for ix in tqdm(range(len(masks_path))):
        mask = np.array(Image.open(masks_path[ix]))[..., :3]
        new_mask = _reconstruct_semantic_mask(mask, contour_color,
                                              class_color_dict, version)
        imsave(dataset_path + target_dir + '/' + img_names[ix] + '.png',
               new_mask)

    print('All masks have been generated at ' + dataset_path + target_dir)

def _semantic_to_instance(mask, img_name, target_path,
                          class_color_dict={1: [255, 105, 180],
                                            0: [0, 0, 255]}):
    """
    Generates instance masks from semantic mask
    By convention the name of the file is the following:
    'img_name.mask_ix.class_ix.png'

    Input
        - mask: np.array of size (H, W, 3)
        - img_name: str, name of the image
        - target_path: str, dir path to save instance mask
        - class_color_dict: dict
    """
    if target_path[-1] != '/':
        target_path += '/'
    red, green, blue = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    for i in class_color_dict.keys():
        class_id = i
        class_mask = (red == class_color_dict[i][0]) &\
                     (green == class_color_dict[i][1]) &\
                     (blue == class_color_dict[i][2])
        labels = label(class_mask)
        for inst_ix in np.unique(labels)[1:]:
            indices = np.where(labels == inst_ix)
            inst_mask = np.zeros(labels.shape)
            inst_mask[indices[0], indices[1]] = 1
            imsave(target_path + img_name + '.' + str(inst_ix) + '.' +
                   str(class_id) + '.png', inst_mask)


def generate_instances(dataset_path, semantic_dir, target_dir,
                       class_color_dict={1: [255, 105, 180],
                                         0: [0, 0, 255]}):
    """
    Generates instance masks from eroded & dilated semantic masks

    Input:
        - dataset_path: str 
        - semantic_dir: str
        - target_dir: str
        - class_color_dict: dict of color & class_id {class_id: [R, G, B]}
    """
    if dataset_path[-1] != '/':
        dataset_path += '/'
    target_path = dataset_path + target_dir
    img_names = [file.split('.', 1)[0] for file in
                 os.listdir(dataset_path + semantic_dir)]
    masks_path = [dataset_path + semantic_dir + '/' + file + '.png'
                  for file in img_names]
    for ix in tqdm(range(len(masks_path))):
        img_name = img_names[ix]
        mask = np.array(Image.open(masks_path[ix]))[..., :3]
        _semantic_to_instance(mask, img_name, target_path, class_color_dict)

    print('All instances have been generated at ' + target_path)