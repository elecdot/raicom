from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from models import baseline_cnn


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    labels: tuple[str, ...]
    image_size: int
    build_model: Callable[[], nn.Module]


MODEL_CANDIDATES = {
    baseline_cnn.NAME: ModelCandidate(
        name=baseline_cnn.NAME,
        labels=baseline_cnn.LABELS,
        image_size=baseline_cnn.IMAGE_SIZE,
        build_model=baseline_cnn.build_model,
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
