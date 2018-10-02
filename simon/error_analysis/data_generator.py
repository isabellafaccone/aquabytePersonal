import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    """
    Load a dataset for target estimation
    
    Args:
       - dataset_name
       - datas_list: datas to keep in the dataset
       - size: float between 0 & 1 (ex 0.7)
    """
    def __init__(self, dataframe, dataset_name,
                 datas_list, size=None, target_list=['length', 'width', 'height', 'volume']):
        self.csv_file = dataframe
        self.dataset_name = dataset_name
        self.datas_list = datas_list
        self.size = size
        self.target_list = target_list
        self.dataset = dataframe
        self.dataset = self.dataset[datas_list]
        if size is not None:
            self.subsample_dataset()
    
    def subsample_dataset(self):
        """
        Subsample dataset to a give size ratio of the whole dataset
        """
        num_examples = self.size
        self.dataset = self.dataset.sample(num_examples).reset_index(drop=True)
        
    def __getitem__(self, index):
        input = {}
        dataset_dir = '/root/data/blender/' + self.dataset_name 
        for data in self.dataset.columns:
            file_name = self.dataset.loc[index, data]
            if '_right' in data : 
                file_path = dataset_dir + '/' + data.replace('_right', '') + '/' + file_name
            elif '_left' in data:
                file_path = dataset_dir + '/' + data.replace('_left', '') + '/' + file_name
            else: 
                file_path = dataset_dir + '/' + data + '/' + file_name
            if 'npy' in file_name:
                if data == 'depth_map':
                    input[data] = np.load(file_path).T
                else:
                    input[data] = np.load(file_path)
            elif 'png' in file_name:
                input[data] = cv2.imread(file_path)
            elif 'json' in file_name:
                label = {}
                if self.target_list is not None:
                    for target in self.target_list:
                        with open(file_path) as f:
                            label[target] = json.load(f)[target]
                else:
                    with open(file_path) as f:
                        label = json.load(f)
        return input, label
    
    def __len__(self):
        return len(self.dataset)