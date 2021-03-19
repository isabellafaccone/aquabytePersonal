import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets import *
import cv2
from functools import partial
import glob
import os
import json

def albumentations_loader(file_path):
    # Read an image with OpenCV
    image = cv2.imread(file_path)

    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class AlbumentationsDatasetFolder(DatasetFolder):

    def __init__(self, root, loader=albumentations_loader, extensions=None, transform=None,
                 target_transform=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform,
                                            loader=loader)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

### COPY/PASTED FROM https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py ###
class AlbumentationsImageFolder(AlbumentationsDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 is_valid_file=None):
        super(AlbumentationsDatasetFolder, self).__init__(
            root,
            loader=albumentations_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform)
        self.imgs = self.samples

def json_loader(json_path):
    try:
        with open(json_path) as inp:
            return json.load(inp)
    except:
        with open(json_path.replace('metadata2', 'metadata')) as inp:
            return json.load(inp)

class FlexibleSkipDataset(VisionDataset):
    def __init__(self, root, transform, target_transform=None, loader=albumentations_loader,
        target_loader=json_loader):
        """Construct from jpgs and metadata"""
        print('initing...')
        self.images = glob.glob(os.path.join(root, '*/**.jpg'))
        self.samples = [
            (ipath,
             ipath.replace('_crop.jpg', '_metadata2.json'))
                     for ipath in self.images]
        self.loader = loader
        self.target_loader = target_loader
        super().__init__(
            root, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, target_path = self.samples[index]
        sample = self.loader(img_path)
        target = self.target_loader(target_path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


BODYPART_COLS = ['HAS_{}'.format(part) for part in [
    'VENTRAL_POSTERIOR', 'VENTRAL_ANTERIOR', 'DORSAL_POSTERIOR', 'DORSAL_ANTERIOR', 'HEAD'
]]
def metadata_to_bodypartmultilabel_repr(metadata):
    if metadata['is_qa_accept']:
        return torch.Tensor([bool(metadata[lab]) for lab in BODYPART_COLS])
    else:
        return torch.Tensor([False]*5)

def metadata_to_bodypart(metadata, bodypart):
    return bool(metadata[bodypart])

class BodypartMultilabelDataset(FlexibleSkipDataset):
    def __init__(self, root, transform, loader=albumentations_loader):
        self.classes = BODYPART_COLS
        super().__init__(root,
            transform, target_transform=metadata_to_bodypartmultilabel_repr, loader=loader)


class BodypartDataset(FlexibleSkipDataset):
    def __init__(self, root, transform, bodypart, loader=albumentations_loader):
        bodypart = 'HAS_{}'.format(bodypart)
        assert bodypart in BODYPART_COLS, (bodypart, BODYPART_COLS)
        tfm = partial(metadata_to_bodypart, bodypart=bodypart)
        self.classes = ['NOT_VISIBLE', 'VISIBLE']
        super().__init__(root,
            transform, target_transform=tfm, loader=loader)
