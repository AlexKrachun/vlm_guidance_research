from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union

import torch
from PIL import Image
import numpy as np

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
PromptLike = Union[str, Sequence[str]]


@dataclass
class ScoreOutput:
    score: torch.Tensor
    loss: torch.Tensor
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseDifferentiableScorer(torch.nn.Module, ABC):
    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__()
        self.device = torch.device(device)

    @abstractmethod
    def forward(
        self,
        image: ImageLike,
        prompt: PromptLike,
        **kwargs: Any,
    ) -> ScoreOutput:
        raise NotImplementedError

    def score(
        self,
        image: ImageLike,
        prompt: PromptLike,
        **kwargs: Any,
    ) -> ScoreOutput:
        return self.forward(image=image, prompt=prompt, **kwargs)
