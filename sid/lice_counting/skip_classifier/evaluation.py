import os

os.environ['CUDA_VISIBL_DEVICES'] = '3'

from model import ImageClassifier
from train import ACCEPT_LABEL, SKIP_LABEL

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

MODEL_PATH = '/root/data/sid/skip_classifier_checkpoints/qa_accept_cogito_skips_03-04-2020/epoch_13/val/'
device = 3
torch.cuda.set_device(device)

model = ImageClassifier([ACCEPT_LABEL, SKIP_LABEL], None, os.path.join(MODEL_PATH, 'model.pt'))
model.to(device)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def albumentations_loader(file_path):
    # Read an image with OpenCV
    image = cv2.imread(file_path)

    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class ImageDataset(Dataset):
    """"""
    def __init__(self, classes, samples, loader=albumentations_loader, extensions=None, transform=None,
          target_transform=None, is_valid_file=None):
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.transform = transform
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = {c: classes.index(c) for c in classes}
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
        index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = albumentations_loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']

        return sample, target

    def __len__(self):
        return len(self.samples)


eval_set = pd.read_csv('/root/data/sid/skip_classifier_datasets/evaluation/naive_eval_set.csv')
classes = [ACCEPT_LABEL, SKIP_LABEL]
paths = eval_set['local_image_path']
labels = eval_set['skip_reasons'].notnull().apply(int)
samples = [(path, label) for path, label in zip(paths, labels)]

from loader import TRANSFORMS

dataset = ImageDataset(classes, samples, transform=TRANSFORMS['pad'])

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
model.to(device)
all_labels = None
all_outputs = None

with torch.no_grad():

    for i, (inputs, tgts) in enumerate(loader):
        tgts = tgts.to(device)
        cuda_inputs = inputs.to(device)
        outputs = model(cuda_inputs)
        #outputs = outputs.cpu().numpy()
        if all_outputs is None:
            all_outputs = outputs
            all_labels = tgts
        else:
            all_outputs = torch.cat((all_outputs, outputs))
            all_labels = torch.cat((all_labels, tgts))
        print(f'Batch: {i}')
