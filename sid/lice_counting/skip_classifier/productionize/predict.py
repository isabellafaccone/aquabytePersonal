import torch
import torch.nn as nn
from torchvision import models
from albumentations import Compose, CenterCrop, LongestMaxSize, PadIfNeeded, Resize, Normalize
from albumentations.pytorch import ToTensor
import cv2


class SkipClassifier(nn.Module):
    def __init__(self):
        super(SkipClassifier, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.out_features
        self.scorer = nn.Linear(num_ftrs, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats = self.model_ft(x)
        scores = self.scorer(feats)
        probs = self.softmax(scores)
        return probs

class SkipPredictor:
    def __init__(self, model_path):
        # Build preprocessor
        self.preprocessor = Compose([
            LongestMaxSize(3000),
            PadIfNeeded(4096, 3000, border_mode=cv2.BORDER_CONSTANT),
            Resize(224, 224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor()
        ])

        # Load model
        self.model = SkipClassifier()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, image_path: str) -> float:
        ### Load the image into a tensor
        image = cv2.imread(image_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessor(image=image)['image']

        ### Predict probability given tensor
        cuda_inputs = torch.unsqueeze(image, dim=0)
        with torch.set_grad_enabled(False):
            outputs = self.model(cuda_inputs)
        cpu_outputs = outputs.detach().numpy()
        accept_confidence = cpu_outputs[0][0]
        return accept_confidence

if __name__ == '__main__':
    # Test the predictor
    MODEL_PATH = '/root/data/sid/needed_data/skip_classifier_checkpoints/qa_accept_cogito_skips_03-04-2020_stratified/epoch_13/val/model.pt'
    predictor = SkipPredictor(MODEL_PATH)
    IMAGE_PATH = '/root/data/sid/needed_data/skip_classifier_datasets/production_evaluation_vikane/images/environment=parallel_PATHSEP_site-id=39_PATHSEP_pen-id=60_PATHSEP_date=2020-03-15_PATHSEP_hour=02_PATHSEP_at=2020-03-15T02:54:27.459824000Z_PATHSEP_left_frame_crop_116_0_4096_2217.jpg'
    print(predictor.predict(IMAGE_PATH))
