import numpy as np
import pandas as pd
import random
import re
import os 
from os.path import dirname

def create_csv(dataset_name,
               datas_list=['stereo_images', 'mask', 'depth_map',
                           'annotations']):
    """
    Creates a dataframe containing all files names for the dataset
    Input:
       - datas_list: list datas from the dataset to include in the
       dataframe
       - dataset_name: name of the dataset
    Output:
        - dataframe object
    """
    dataset_path = dirname(os.getcwd()) + '/data/' + str(dataset_name) + '/'
    regex = re.compile(r'\d+')
    datas = {}
    # Getting the files
    for data_type in datas_list:
        files = os.listdir(dataset_path + data_type)
        if data_type in ['mask', 'stereo_images']:
            right_files = [f for f in files if 'right' in f]
            if 'right' in right_files:
                right_files.remove('right')
            right_id = [int(regex.findall(f)[0]) for f in right_files]
            left_files = [f for f in files if 'left' in f]
            if 'left' in left_files:
                left_files.remove('left')
            left_id = [int(regex.findall(f)[0]) for f in left_files] 
            datas[data_type + '_left'] = (left_files, left_id)
            datas[data_type + '_right'] = (right_files, right_id)
        else:
            files_id = [int(regex.findall(f)[0]) for f in files]
            datas[data_type] = ((files, files_id))
    size = len(datas[datas.keys()[0]][0])
    dataset = pd.DataFrame(index=range(size), columns=datas.keys())
    
    # Let's fill the dataframe now
    for key in datas.keys():
        for ix in range(size):
            dataset[key][datas[key][1][ix]] = datas[key][0][ix]
    dataset.to_csv(dataset_name + '.csv')
    
    return dataset

def split_train_test(df, train_split):
    """
    Split a dataframe randomly into train & test set
    """
    train_ix = random.sample(range(len(df)), int(train_split * len(df)))
    test_ix = list(set(df.index) - set(train_ix))
    train_df = df.iloc[train_ix, :].reset_index(drop=True)
    test_df = df.iloc[test_ix, :].reset_index(drop=True)
    
    return train_df, test_df