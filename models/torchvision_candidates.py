import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


LABELS = ("cloudy", "rainy", "snowy", "sunny")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EFFICIENTNET_B0_NAME = "efficientnet_b0"
EFFICIENTNET_B0_IMAGE_SIZE = 224
EFFICIENTNET_B0_WEIGHTS_NAME = "EfficientNet_B0_Weights.DEFAULT"


def build_efficientnet_b0(pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(LABELS))
    return model
