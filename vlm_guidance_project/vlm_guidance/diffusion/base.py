from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseDiffusionBackend(ABC):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def init_latents(
        self,
        height: int,
        width: int,
        batch_size: int,
        seed: Optional[int],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def timesteps(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_eps_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def scheduler_step(
        self,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
