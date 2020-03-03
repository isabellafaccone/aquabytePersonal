from albumentations import Compose, CenterCrop, LongestMaxSize, PadIfNeeded, Resize, Normalize
from albumentations.pytorch import ToTensor
from albumentations.pytorch.datasets import AlbumentationsImageFolder
import cv2
import torch
import os


DATA_DIR = '/root/data/sid/lice_skip_data/'
NUM_EX = 100000

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
CLASS_COUNTS = dict()
for lab in os.listdir(DATA_DIR):
    CLASS_COUNTS[lab] = (len(os.listdir(os.path.join(DATA_DIR, lab)))/2)

def get_dataloader(transform, bsz, split_size):
    print('Loading dataset from directory...')
    dataset = AlbumentationsImageFolder(DATA_DIR, transform)
    print('Splitting into folds...')
    datasets = dict()
    val_size = (1 - split_size) / 2
    sizes = [int(NUM_EX*split_size), int(NUM_EX * val_size)]
    sizes.append(NUM_EX - sum(sizes))
    datasets['train'], datasets['val'], datasets['test'] = torch.utils.data.random_split(dataset, sizes)
    print('Building dataloaders...')
    dataloaders = {split: torch.utils.data.DataLoader(datasets[split], batch_size=bsz, shuffle=True, num_workers=4)
                   for split in datasets}
    return dataloaders, dataset.classes

if __name__ == '__main__':
    get_dataloader(CENTER_CROP_TRANSFORM)
    get_dataloader(PAD_TRANSFORM)
