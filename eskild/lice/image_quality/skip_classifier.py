import json
import logging

from albumentations import Compose, LongestMaxSize, PadIfNeeded, Resize, Normalize
from albumentations.pytorch import ToTensor
import cv2
import torch
import torch.nn as nn
from torchvision import models

from . import normalize


class SkipClassifier(nn.Module):
    def __init__(self):
        super(SkipClassifier, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.out_features
        self.scorer = nn.Linear(num_ftrs, 2)
        self.softmax = nn.Softmax(dim=1)
        if torch.cuda.is_available():
            self.cuda()

    # pylint: disable=W0221
    def forward(self, x):
        feats = self.model_ft(x)
        scores = self.scorer(feats)
        probs = self.softmax(scores)
        return probs


class SkipPredictor:

    def __init__(self, model_path, normalization_param_path=None):
        # Build preprocessor
        self.preprocessor = Compose([
            LongestMaxSize(3000),
            PadIfNeeded(4096, 3000, border_mode=cv2.BORDER_CONSTANT),
            Resize(224, 224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])

        # Load model
        logging.info(f"SkipPredictor loading - {model_path}")
        self.model = SkipClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        self.model.eval()

        logging.info(f"SkipPredictor load_normalizer - {normalization_param_path}")
        self.normalizer = self.load_normalizer(normalization_param_path)


    def load_normalizer(self, normalization_param_path):
        if normalization_param_path:
            with open(normalization_param_path) as fp:
                normalize_params = json.load(fp)
        else:
            normalize_params = None
        return normalize.Normalizer(normalize_params)


    def predict(self, image) -> float:

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessor(image=image)['image']

        ### Predict probability given tensor
        cuda_inputs = torch.unsqueeze(image, dim=0).cuda()
        with torch.set_grad_enabled(False):
            outputs = self.model(cuda_inputs)
        cpu_outputs = outputs.cpu().detach().numpy()
        accept_confidence = cpu_outputs[0][0]
        return accept_confidence.item()


    def predict_and_normalize(self, image, pen_id) -> float:
        score = self.predict(image)
        return self.normalizer.getNormalizedScore(score, pen_id)
