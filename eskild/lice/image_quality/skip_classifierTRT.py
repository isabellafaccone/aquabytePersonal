import json
import logging

from albumentations import Compose, LongestMaxSize, PadIfNeeded, Resize, Normalize
from albumentations.pytorch import ToTensor
import cv2
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import numpy as np
from torch2trt import torch2trt
from torchvision import models
from datetime import datetime



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
        
        logging.info("Converting model to TensorRT 32...")
        x = torch.ones(64,3,224,224).cuda()
        self.modelTRT_32 = torch2trt(self.model, [x], fp16_mode=False, log_level=trt.Logger.VERBOSE)
        logging.info("Converting model to TensorRT 16...")
        self.modelTRT_16 = torch2trt(self.model, [x], fp16_mode=True, log_level=trt.Logger.VERBOSE)
        
        logging.info(f"SkipPredictor load_normalizer - {normalization_param_path}")
        self.normalizer = self.load_normalizer(normalization_param_path)


    def load_normalizer(self, normalization_param_path):
        if normalization_param_path:
            with open(normalization_param_path) as fp:
                normalize_params = json.load(fp)
        else:
            normalize_params = None
        return normalize.Normalizer(normalize_params)


    def predict_trt_32(self, cuda_inputs):
        start_t = datetime.now()
        with torch.set_grad_enabled(False):
            outputs = self.modelTRT_32(cuda_inputs)
        torch.cuda.current_stream().synchronize()
        cpu_outputs = outputs.cpu().detach().numpy()
        accept_confidence = cpu_outputs[0][0]
        t = (datetime.now() - start_t).total_seconds()
        return accept_confidence.item(), t
    
    def predict_trt_16(self, cuda_inputs):
        start_t = datetime.now()
        with torch.set_grad_enabled(False):
            outputs = self.modelTRT_16(cuda_inputs)
        torch.cuda.current_stream().synchronize()
        cpu_outputs = outputs.cpu().detach().numpy()
        accept_confidence = cpu_outputs[0][0]
        t = (datetime.now() - start_t).total_seconds()
        return accept_confidence.item(), t
    
    def predict_tc(self, cuda_inputs):
        start_t = datetime.now()
        with torch.set_grad_enabled(False):
            outputs = self.model(cuda_inputs)
        torch.cuda.current_stream().synchronize()
        cpu_outputs = outputs.cpu().detach().numpy()
        accept_confidence = cpu_outputs[0][0]
        t = (datetime.now() - start_t).total_seconds()
        return accept_confidence.item(), t   

    def predict(self, image, ix,key, state):

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessor(image=image)['image']

        ### Predict probability given tensor
        cuda_inputs = torch.unsqueeze(image.contiguous(), dim=0).cuda()
        torch.cuda.current_stream().synchronize()
        
        score_t32, t_t32 = self.predict_trt_32(cuda_inputs)
        score_t16, t_t16 = self.predict_trt_16(cuda_inputs)
        score_p, t_p = self.predict_tc(cuda_inputs)
        
        return {
            'ix':ix,
            'key':key,
            'state':state,
            'trt_32':{
                'score':score_t32,
                'diff':np.abs(score_t32-score_p),
                'time':t_t32
            },
            'trt_16':{
                'score':score_t16,
                'diff':np.abs(score_t16-score_p),
                'time':t_t16
            },
            'torch':{
                'score':score_p,
                'time':t_p
            }
        }    
        


    def predict_and_normalize(self, image, pen_id) -> float:
        score = self.predict(image)
        return self.normalizer.getNormalizedScore(score, pen_id)
