from __future__ import annotations

from typing import List, Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

from vlm_guidance.generation.base import BaseText2ImagePipeline


_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class Flux1DevPipeline(BaseText2ImagePipeline):
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        enable_model_cpu_offload: bool = True,
        max_sequence_length: int = 512,
    ) -> None:
        requested_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            torch_dtype = "float32"
            enable_model_cpu_offload = False
        self._device = torch.device(device)
        self.max_sequence_length = max_sequence_length
        dtype = _DTYPES[torch_dtype.lower()]
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if enable_model_cpu_offload and requested_device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self._device)

    @property
    def pipeline_name(self) -> str:
        return "flux1_dev"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        # FLUX docs commonly use a CPU generator even when models are offloaded to GPU.
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        out = self.pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
        )
        return list(out.images)
