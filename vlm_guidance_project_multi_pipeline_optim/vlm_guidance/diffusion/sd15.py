from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

from vlm_guidance.diffusion.base import BaseDiffusionBackend
from vlm_guidance.utils.amp import PrecisionConfig, autocast_context


@dataclass
class SD15Config:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda:0"
    weights_dtype: str = "float16"
    autocast: PrecisionConfig = PrecisionConfig(enabled=False, dtype="bfloat16")


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


class StableDiffusion15Backend(BaseDiffusionBackend):
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda:0",
        weights_dtype: str = "float16",
        autocast_enabled: bool = False,
        autocast_dtype: str = "bfloat16",
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu" and device.startswith("cuda"):
            weights_dtype = "float32"
            autocast_enabled = False
        self._dtype = _resolve_dtype(weights_dtype)
        self.precision = PrecisionConfig(enabled=autocast_enabled, dtype=autocast_dtype)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self._dtype).to(self._device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self._dtype).to(self._device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self._dtype).to(self._device)
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        for module in (self.text_encoder, self.vae, self.unet):
            for p in module.parameters():
                p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = "", max_length: int = 77) -> torch.Tensor:
        text_inputs = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        uncond_inputs = self.tokenizer(negative_prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        cond_embeds = self.text_encoder(text_inputs.input_ids.to(self._device))[0]
        uncond_embeds = self.text_encoder(uncond_inputs.input_ids.to(self._device))[0]
        return torch.cat([uncond_embeds, cond_embeds], dim=0)

    def init_latents(self, height: int, width: int, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        latent_shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        latents = randn_tensor(latent_shape, generator=generator, device=self._device, dtype=self._dtype)
        return latents * self.scheduler.init_noise_sigma

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.scheduler.set_timesteps(num_inference_steps, device=self._device)

    def predict_eps_with_cfg(self, x_t: torch.Tensor, t: torch.Tensor, text_embeds: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        latent_model_input = torch.cat([x_t] * 2, dim=0)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        with autocast_context(self._device, self.precision):
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return eps_pred

    def predict_x0_from_eps(self, x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_idx = int(t.item()) if hasattr(t, "item") else int(t)
        alpha_bar_t = self.scheduler.alphas_cumprod[t_idx].to(device=x_t.device, dtype=x_t.dtype)
        while alpha_bar_t.ndim < x_t.ndim:
            alpha_bar_t = alpha_bar_t.view(*alpha_bar_t.shape, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        return (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents / 0.18215).to(device=self.vae.device, dtype=next(self.vae.parameters()).dtype)
        with autocast_context(self._device, self.precision):
            images = self.vae.decode(latents).sample
        return (images / 2 + 0.5).clamp(0, 1).float()

    @torch.no_grad()
    def scheduler_step(self, eps_pred: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.scheduler.step(eps_pred, t, x_t).prev_sample
