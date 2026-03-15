from __future__ import annotations

from typing import Dict, Any

import torch


def tensor_stats(name: str, x: torch.Tensor) -> Dict[str, Any]:
    finite = torch.isfinite(x)
    x_det = x.detach()
    return {
        "name": name,
        "ok": bool(finite.all().item()),
        "dtype": str(x.dtype),
        "shape": tuple(x.shape),
        "min": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
        "max": float(torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
        "has_nan": bool(torch.isnan(x_det).any().item()),
        "has_inf": bool(torch.isinf(x_det).any().item()),
    }
