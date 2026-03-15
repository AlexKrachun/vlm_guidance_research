from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class PrecisionConfig:
    enabled: bool = False
    dtype: str = "bfloat16"

    def resolved_dtype(self) -> Optional[torch.dtype]:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = self.dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported autocast dtype: {self.dtype}")
        return mapping[key]


def autocast_context(device: torch.device, precision: PrecisionConfig):
    if not precision.enabled:
        return contextlib.nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=precision.resolved_dtype())
