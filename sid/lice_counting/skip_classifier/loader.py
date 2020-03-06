from albumentations import Compose, CenterCrop, LongestMaxSize, PadIfNeeded, Resize, Normalize
from albumentations.pytorch import ToTensor
from image_folder import AlbumentationsImageFolder
import cv2
import torch
import os
from data import MODEL_DATA_PATH

DATA_FNAME = 'qa_accept_cogito_skips_03-04-2020/images'

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


def get_dataloader(fname, transform, bsz, split_size):
    data_dir = os.path.join(MODEL_DATA_PATH, fname, 'images')
    print('Loading dataset from directory...')
    dataset = AlbumentationsImageFolder(data_dir, transform)
    print('Splitting into folds...')
    datasets = dict()
    val_size = (1 - split_size) / 2
    num_ex = len(dataset.samples)
    sizes = [int(num_ex*split_size), int(num_ex * val_size)]
    sizes.append(num_ex - sum(sizes))
    train, val, test = torch.utils.data.random_split(dataset, sizes)
    datasets = {
        'train': train,
        'val': val,
        'test': test
    }
    print('Building dataloaders...')
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=bsz, shuffle=True, num_workers=4)
        for split in datasets
    }
    ### Calculate Class Distribution ###
    class_counts = dict()
    labels = os.listdir(data_dir)
    for lab in labels:
        class_counts[lab] = (len(os.listdir(os.path.join(data_dir, lab)))/2)
    return dataloaders, dataset.classes, class_counts

if __name__ == '__main__':
    get_dataloader(TRANSFORMS['center_crop'], 10, 0.8)
    get_dataloader(TRANSFORMS['pad'], 10, 0.8)
