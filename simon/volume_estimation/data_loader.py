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
        num_examples = int(self.size * len(self.dataset))
        self.dataset = self.dataset.sample(num_examples).reset_index(drop=True)
        
    def __getitem__(self, index):
        input = {}
        dataset_dir = '/root/data/' + self.dataset_name 
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
    
class CroperNormalizer():
    """
    Normalizes depth map masks it by the fish mask 
    """
    def __init__(self, target_img_size=(300, 100)):
        self.target_img_size = target_img_size
    
    def get_bb_from_mask(self, mask):
        """
        Computes the bounding box coordinates from mask
        """
        x_end, x_start = np.where(mask == 1)[0].max(), np.where(mask == 1)[0].min()
        y_end, y_start = np.where(mask == 1)[1].max(), np.where(mask == 1)[1].min()
        
        return (x_start, x_end, y_start, y_end)
    
    def normalize_dmap(self, dmap):
        """
        Normalize depth map to born its values between 0 & 1
        """
        M, m = dmap.max(), dmap.min()
        normalized_dmap = (dmap -  m) / (M - m )
        
        return normalized_dmap
    
    def process_dmap(self, data):
        """
        Apply the transformation for a batch
        
        Input:
           - data: (input, label) output of DataGenerator class
           
        Output:
            - label_data : np.array of size (batch_size, 2)
            - input_data : np.array of size (batch_size, target_img_size)
        """
        input_data = np.zeros((self.target_img_size[1],
                              self.target_img_size[0]))
        dmap, mask = data[0]['depth_map'], data[0]['mask_left']
        label_data = data[1]
        x_start, x_end, y_start, y_end = self.get_bb_from_mask(mask)
        croped_dmap = dmap[x_start:x_end, y_start:y_end]
        croped_mask = mask[x_start:x_end, y_start:y_end]
        normalized_dmap = self.normalize_dmap(croped_dmap)
        # Mask the normalized_dmap
        normalized_dmap[np.where(croped_mask == 0)] = 1
        # Resize the normalized_dmap
        resized_dmap = cv2.resize(normalized_dmap, self.target_img_size)
        # Fill the numpy arrays
        input_data = resized_dmap
        
        return input_data, label_data