from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]


def to_bchw_float01(image: ImageLike) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim != 3:
            raise ValueError("NumPy image must have shape [H, W, C] or [H, W].")
        image = torch.from_numpy(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be PIL.Image, np.ndarray or torch.Tensor")

    x = image
    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)
    elif x.ndim == 3:
        if x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported image ndim={x.ndim}")

    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def save_image_tensor(images_bchw: torch.Tensor, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = images_bchw[0].detach().cpu().clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


@torch.no_grad()
def save_diff_image(img_a: torch.Tensor, img_b: torch.Tensor, path: Union[str, Path], amplify: float = 4.0) -> None:
    diff = (img_b - img_a).abs()
    diff = (diff / diff.max().clamp(min=1e-8)) * amplify
    diff = diff.clamp(0, 1)
    save_image_tensor(diff, path)
