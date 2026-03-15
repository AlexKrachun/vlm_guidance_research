from __future__ import annotations

from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
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


class VanillaSD15Pipeline(BaseText2ImagePipeline):
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        enable_attention_slicing: bool = False,
        enable_xformers_memory_efficient_attention: bool = False,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            torch_dtype = "float32"
        self._device = torch.device(device)
        dtype = _DTYPES[torch_dtype.lower()]
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe = self.pipe.to(self._device)
        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        if enable_xformers_memory_efficient_attention:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @property
    def pipeline_name(self) -> str:
        return "vanilla_sd15"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        out = self.pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return list(out.images)
