from albumentations import Compose, CenterCrop, LongestMaxSize, PadIfNeeded, Resize, Normalize
import pandas as pd
from albumentations.pytorch import ToTensor
from image_folder_new import AlbumentationsImageFolder, BodypartMultilabelDataset, BodypartDataset, BODYPART_COLS
import cv2
import torch
import os
import json

from config import SKIP_CLASSIFIER_DATASET_DIRECTORY, SKIP_CLASSIFIER_IMAGE_DIRECTORY
import pickle
from tqdm import tqdm

DATA_FNAME = 'qa_accept_cogito_skips_03-04-2020/images'
TRAIN_TEST_SPLIT_PATH = '/root/data/sid/needed_data/skip_classifier_datasets/splits'

### Image Data Augmentation ###

TRANSFORMS = {
    'center_crop': Compose([
        CenterCrop(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ]),
    'pad': Compose([
        LongestMaxSize(3000),
        # Using Border constant for padding because partial fish that are not in the center of the crop can look like full fish
        PadIfNeeded(4096, 3000, border_mode=cv2.BORDER_CONSTANT),
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])
}

def get_multlabel_class_counts(fname, bodypart=None):
    sampled = pd.read_csv(os.path.join(SKIP_CLASSIFIER_DATASET_DIRECTORY, fname + '.csv'))
    ratios = (sampled[BODYPART_COLS].sum() / len(sampled))
    if bodypart is None:
        return (1 / ratios).to_list()
    else:
        return (1 / ratios['HAS_' + bodypart])


def get_dataloader(fname, transform, bsz, split_size, model_type='full_fish'):
    data_dir = os.path.join(SKIP_CLASSIFIER_IMAGE_DIRECTORY, fname)
    print('Loading dataset from directory...')
    if model_type == 'full_fish':
        dataset = AlbumentationsImageFolder(data_dir, transform)
        ### Calculate Class Distribution ###
        class_counts = dict()
        labels = os.listdir(data_dir)
        for lab in labels:
            class_counts[lab] = (len(os.listdir(os.path.join(data_dir, lab)))/2)

    elif model_type == 'bodyparts':
        print('Using Bodyparts Data loader...')
        dataset = BodypartMultilabelDataset(data_dir, transform)
        class_counts = get_multlabel_class_counts(fname)
    elif model_type.startswith('single_bodypart_'):
        bodypart = model_type[len('single_bodypart_'):]
        assert 'HAS_' + bodypart in BODYPART_COLS
        dataset = BodypartDataset(data_dir, transform, bodypart)
        class_counts = get_multlabel_class_counts(fname, bodypart=bodypart)
    else:
        print('model type must be bodyparts or full_fish')
        print(model_type)
        assert False

    print(dataset[0])
    print('Splitting into folds...')
    datasets = dict()
    val_size = (1 - split_size) / 2
    num_ex = len(dataset)
    sizes = [int(num_ex*split_size), int(num_ex * val_size)]
    sizes.append(num_ex - sum(sizes))
    train, val, test = torch.utils.data.random_split(dataset, sizes)
    dataset_splits = {
        'original': list(dataset.samples),
        'train_indices': list(train.indices),
        'val_indices': list(val.indices),
        'test_indices': list(test.indices)
    }

    for idx in range(10):
       ex, labs = train[idx]
       print(labs)
    print({k:[type(v), v[:10]] for k,v in dataset_splits.items()})
    split_path = os.path.join(TRAIN_TEST_SPLIT_PATH, f'{fname}_splits.json')
    with open(split_path, 'w') as f:
        json.dump(dataset_splits, f)

    datasets = {
        'train': train,
        'val': val,
        'test': test
    }
    print('Building dataloaders...')
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=bsz, shuffle=False, num_workers=4)
        for split in datasets
    }
    return dataloaders, dataset.classes, class_counts

if __name__ == '__main__':
    bsz = 10
    #get_dataloader(TRANSFORMS['center_crop'], 10, 0.8)
    #loader, _, _ = get_dataloader('09-23-2020_bodypart', TRANSFORMS['pad'], bsz, 0.8, model_type='full_fish')
    #for inputs, labels in loader['train']:
    #    print(inputs.shape)
    #    print(labels)
    #    break
    loader, _, _ = get_dataloader('09-23-2020_bodypart', TRANSFORMS['pad'], bsz, 0.8, model_type='bodyparts')
    for inputs, labels in loader['train']:
        #labs = torch.cat(labels).unsqueeze(1).transpose(0, 1).view(bsz, 5)
        print(inputs.shape)
        #print(labs.shape)
        print(labels)
        #print(labs)
        break
