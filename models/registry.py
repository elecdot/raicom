from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

from models import baseline_cnn, torchvision_candidates


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    labels: tuple[str, ...]
    image_size: int
    build_model: Callable[[bool], nn.Module]
    pretrained_weights_name: Optional[str] = None
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolation: str = "bilinear"


MODEL_CANDIDATES = {
    baseline_cnn.NAME: ModelCandidate(
        name=baseline_cnn.NAME,
        labels=baseline_cnn.LABELS,
        image_size=baseline_cnn.IMAGE_SIZE,
        build_model=baseline_cnn.build_model,
    ),
    torchvision_candidates.EFFICIENTNET_B0_NAME: ModelCandidate(
        name=torchvision_candidates.EFFICIENTNET_B0_NAME,
        labels=torchvision_candidates.LABELS,
        image_size=torchvision_candidates.EFFICIENTNET_B0_IMAGE_SIZE,
        build_model=torchvision_candidates.build_efficientnet_b0,
        pretrained_weights_name=torchvision_candidates.EFFICIENTNET_B0_WEIGHTS_NAME,
        mean=torchvision_candidates.IMAGENET_MEAN,
        std=torchvision_candidates.IMAGENET_STD,
        interpolation="bicubic",
    ),
    torchvision_candidates.EFFICIENTNET_B3_NAME: ModelCandidate(
        name=torchvision_candidates.EFFICIENTNET_B3_NAME,
        labels=torchvision_candidates.LABELS,
        image_size=torchvision_candidates.EFFICIENTNET_B3_IMAGE_SIZE,
        build_model=torchvision_candidates.build_efficientnet_b3,
        pretrained_weights_name=torchvision_candidates.EFFICIENTNET_B3_WEIGHTS_NAME,
        mean=torchvision_candidates.IMAGENET_MEAN,
        std=torchvision_candidates.IMAGENET_STD,
        interpolation="bicubic",
    ),
}


def available_model_candidates():
    return tuple(sorted(MODEL_CANDIDATES))


def get_model_candidate(name):
    try:
        return MODEL_CANDIDATES[name]
    except KeyError as exc:
        choices = ", ".join(available_model_candidates())
        raise ValueError(
            f"unknown model candidate {name!r}; choose one of: {choices}"
        ) from exc
