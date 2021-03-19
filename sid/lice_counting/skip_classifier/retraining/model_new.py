import torch
import torch.nn as nn
from torchvision import models

class ImageClassifier(nn.Module):
    def __init__(self, class_names, savename, state_dict_path=None):
        super(ImageClassifier, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        if state_dict_path is not None:
            print(f'Loading from {state_dict_path}')
            self.model_ft.load_state_dict(torch.load(state_dict_path))
        num_ftrs = self.model_ft.fc.out_features
        self.scorer = nn.Linear(num_ftrs, len(class_names))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = self.model_ft(x)
        scores = self.scorer(feats)
        probs = self.softmax(scores)
        return probs


class MultilabelClassifier(nn.Module):
    def __init__(self, savename, num_labels, state_dict_path=None):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=True)
        if state_dict_path is not None:
            print(f'Loading from {state_dict_path}')
            self.model_ft.load_state_dict(torch.load(state_dict_path))
        num_ftrs = self.model_ft.fc.out_features
        self.scorer = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        feats = self.model_ft(x)
        scores = self.scorer(feats)
        return scores
