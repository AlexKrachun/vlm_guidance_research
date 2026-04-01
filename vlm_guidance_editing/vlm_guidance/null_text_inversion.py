from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    F = None
    DDIMScheduler = None
    StableDiffusionPipeline = None

if torch is not None:
    no_grad = torch.no_grad
else:  # pragma: no cover - keeps module importable without ML deps
    def no_grad():
        def decorator(fn):
            return fn
        return decorator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def require_ml_deps() -> None:
    if torch is None or DDIMScheduler is None or StableDiffusionPipeline is None:
        raise ImportError(
            "This module requires torch, diffusers, and transformers. "
            "Install them in the active environment before running the pipeline."
        )


def resolve_dtype(name: str) -> "torch.dtype":
    require_ml_deps()
    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(lookup)}")
    return lookup[key]


def load_sample(sample_dir: Path) -> Tuple[Path, str, Optional[str]]:
    sample_dir = Path(sample_dir)
    image_path = next((p for p in sorted(sample_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS), None)
    if image_path is None:
        raise FileNotFoundError(f"No image file found in {sample_dir}")

    source_prompt_path = sample_dir / "source_prompt.txt"
    if not source_prompt_path.exists():
        raise FileNotFoundError(f"Missing source prompt: {source_prompt_path}")
    source_prompt = source_prompt_path.read_text().strip()

    edit_prompt_path = sample_dir / "edit_prompt.txt"
    edit_prompt = edit_prompt_path.read_text().strip() if edit_prompt_path.exists() else None
    return image_path, source_prompt, edit_prompt


def discover_sample_dirs(samples_root: Path, limit: Optional[int] = None) -> List[Path]:
    samples_root = Path(samples_root)
    sample_dirs = [p for p in sorted(samples_root.iterdir()) if p.is_dir()]
    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    return sample_dirs


def center_crop_and_resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize(size, Image.Resampling.LANCZOS)


def load_image(image_path: Path, size: Optional[Tuple[int, int]] = (512, 512)) -> "torch.Tensor":
    require_ml_deps()
    image = Image.open(image_path).convert("RGB")
    if size is not None:
        image = center_crop_and_resize(image, size)
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.clamp(0.0, 1.0)


@dataclass
class InversionArtifacts:
    source_prompt: str
    edit_prompt: Optional[str]
    image_latent: "torch.Tensor"
    ddim_latents: List["torch.Tensor"]
    null_text_embeddings: List["torch.Tensor"]
    null_text_loss_history: List[float]
    reconstruction: "torch.Tensor"
    edited: Optional["torch.Tensor"]


class NullTextInversionPipeline:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda:0",
        weights_dtype: str = "float32",
        num_ddim_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> None:
        require_ml_deps()
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = torch.device(device)
        self.weights_dtype = resolve_dtype(weights_dtype)
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.weights_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(self.num_ddim_steps, device=self.device)
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        for module in (self.text_encoder, self.vae, self.unet):
            for param in module.parameters():
                param.requires_grad_(False)

    @no_grad()
    def encode_text(self, prompt: str) -> "torch.Tensor":
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(text_input.input_ids.to(self.device))[0]

    @no_grad()
    def encode_image(self, image_bchw: "torch.Tensor") -> "torch.Tensor":
        image_bchw = image_bchw.to(self.device, dtype=self.weights_dtype)
        image_bchw = image_bchw * 2.0 - 1.0
        posterior = self.vae.encode(image_bchw).latent_dist
        return posterior.mean * self.latent_scaling_factor

    @no_grad()
    def decode_latents(self, latents: "torch.Tensor") -> "torch.Tensor":
        latents = latents.to(self.device, dtype=self.weights_dtype)
        image = self.vae.decode(latents / self.latent_scaling_factor).sample
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
        return image.float().cpu()

    def decode_latents_differentiable(self, latents: "torch.Tensor") -> "torch.Tensor":
        latents = latents.to(self.device, dtype=self.weights_dtype)
        image = self.vae.decode(latents / self.latent_scaling_factor).sample
        return (image / 2.0 + 0.5).clamp(0.0, 1.0).float()

    def predict_noise_single(
        self,
        latents: "torch.Tensor",
        timestep: "torch.Tensor",
        text_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        latents = latents.to(self.device, dtype=self.weights_dtype)
        text_embeddings = text_embeddings.to(self.device, dtype=self.weights_dtype)
        return self.unet(latents, timestep, encoder_hidden_states=text_embeddings).sample

    def predict_noise_cfg(
        self,
        latents: "torch.Tensor",
        timestep: "torch.Tensor",
        uncond_embeddings: "torch.Tensor",
        cond_embeddings: "torch.Tensor",
    ) -> "torch.Tensor":
        latents_input = torch.cat([latents, latents], dim=0).to(self.device, dtype=self.weights_dtype)
        context = torch.cat(
            [
                uncond_embeddings.to(self.device, dtype=self.weights_dtype),
                cond_embeddings.to(self.device, dtype=self.weights_dtype),
            ],
            dim=0,
        )
        noise_pred = self.unet(latents_input, timestep, encoder_hidden_states=context).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        return noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

    def prev_step(self, model_output: "torch.Tensor", timestep: int, sample: "torch.Tensor") -> "torch.Tensor":
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device, dtype=sample.dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep].to(sample.device, dtype=sample.dtype)
        else:
            alpha_prod_t_prev = self.scheduler.final_alpha_cumprod.to(sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: "torch.Tensor", timestep: int, sample: "torch.Tensor") -> "torch.Tensor":
        timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps,
            self.scheduler.config.num_train_timesteps - 1,
        )
        next_timestep = timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        if timestep >= 0:
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device, dtype=sample.dtype)
        else:
            alpha_prod_t = self.scheduler.final_alpha_cumprod.to(sample.device, dtype=sample.dtype)
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep].to(sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        next_sample_direction = (1 - alpha_prod_t_next).sqrt() * model_output
        next_sample = alpha_prod_t_next.sqrt() * next_original_sample + next_sample_direction
        return next_sample

    def predict_x0_from_eps(self, x_t: "torch.Tensor", eps_pred: "torch.Tensor", timestep: int) -> "torch.Tensor":
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(x_t.device, dtype=x_t.dtype)
        beta_prod_t = 1 - alpha_prod_t
        return (x_t - beta_prod_t.sqrt() * eps_pred) / alpha_prod_t.sqrt()

    def _prepare_grad_for_update(
        self,
        grad: "torch.Tensor",
        normalize_grad: bool = True,
        clamp_grad_value: Optional[float] = 1.0,
    ) -> Tuple["torch.Tensor", float, float, bool]:
        grad_fp32 = grad.detach().float()
        raw_finite = torch.isfinite(grad_fp32).all()
        raw_norm = torch.linalg.vector_norm(grad_fp32).detach()

        if normalize_grad and raw_finite and torch.isfinite(raw_norm) and raw_norm.item() > 0:
            grad_fp32 = grad_fp32 / raw_norm

        if clamp_grad_value is not None:
            grad_fp32 = grad_fp32.clamp(-clamp_grad_value, clamp_grad_value)

        processed_finite = torch.isfinite(grad_fp32).all()
        processed_norm = torch.linalg.vector_norm(grad_fp32).detach()
        is_valid = bool(raw_finite.item() and processed_finite.item() and torch.isfinite(raw_norm).item() and torch.isfinite(processed_norm).item())
        return grad_fp32.to(dtype=grad.dtype), float(raw_norm.item()), float(processed_norm.item()), is_valid

    @no_grad()
    def ddim_inversion(self, image_latent: "torch.Tensor", cond_embeddings: "torch.Tensor") -> List["torch.Tensor"]:
        ddim_latents = [image_latent.detach().clone()]
        latent = image_latent.detach().clone().to(self.device, dtype=self.weights_dtype)
        reversed_timesteps = list(reversed(self.scheduler.timesteps))
        for timestep in reversed_timesteps:
            noise_pred = self.predict_noise_single(latent, timestep, cond_embeddings)
            latent = self.next_step(noise_pred, int(timestep.item()), latent)
            ddim_latents.append(latent.detach().clone())
        return [latent.float().cpu() for latent in ddim_latents]

    def null_text_optimization(
        self,
        ddim_latents: Sequence["torch.Tensor"],
        cond_embeddings: "torch.Tensor",
        base_uncond_embeddings: "torch.Tensor",
        num_inner_steps: int,
        lr: float,
        early_stop_eps: float,
    ) -> Tuple[List["torch.Tensor"], List[float]]:
        ddim_latents = [latent.to(self.device, dtype=self.weights_dtype) for latent in ddim_latents]
        cond_embeddings = cond_embeddings.to(self.device, dtype=self.weights_dtype)
        base_uncond_embeddings = base_uncond_embeddings.to(self.device, dtype=torch.float32)

        optimized_embeddings: List[torch.Tensor] = []
        loss_history: List[float] = []
        latent_cur = ddim_latents[-1]

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            target_prev = ddim_latents[-step_index - 2]
            uncond_embeddings = base_uncond_embeddings.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([uncond_embeddings], lr=lr * (1.0 - step_index / 100.0))
            with torch.no_grad():
                noise_pred_cond = self.predict_noise_single(latent_cur, timestep, cond_embeddings)

            best_loss = float("inf")
            for _ in range(num_inner_steps):
                noise_pred_uncond = self.predict_noise_single(
                    latent_cur,
                    timestep,
                    uncond_embeddings.to(self.device, dtype=self.weights_dtype),
                )
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev_rec = self.prev_step(noise_pred, int(timestep.item()), latent_cur)
                loss = F.mse_loss(latent_prev_rec.float(), target_prev.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                best_loss = min(best_loss, float(loss.item()))
                if best_loss < early_stop_eps + step_index * 2e-5:
                    break

            optimized_step = uncond_embeddings[:1].detach().to(self.device, dtype=self.weights_dtype)
            optimized_embeddings.append(optimized_step.float().cpu())
            loss_history.append(best_loss)

            with torch.no_grad():
                noise_pred = self.predict_noise_cfg(latent_cur, timestep, optimized_step, cond_embeddings)
                latent_cur = self.prev_step(noise_pred, int(timestep.item()), latent_cur)

        return optimized_embeddings, loss_history

    @no_grad()
    def sample_with_uncond_sequence(
        self,
        start_latent: "torch.Tensor",
        cond_embeddings: "torch.Tensor",
        uncond_embeddings_sequence: Sequence["torch.Tensor"],
    ) -> "torch.Tensor":
        latents = start_latent.to(self.device, dtype=self.weights_dtype)
        cond_embeddings = cond_embeddings.to(self.device, dtype=self.weights_dtype)

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            uncond_embeddings = uncond_embeddings_sequence[step_index].to(self.device, dtype=self.weights_dtype)
            noise_pred = self.predict_noise_cfg(latents, timestep, uncond_embeddings, cond_embeddings)
            latents = self.prev_step(noise_pred, int(timestep.item()), latents)
        return latents.float().cpu()

    def sample_with_vlm_guidance(
        self,
        start_latent: "torch.Tensor",
        cond_embeddings: "torch.Tensor",
        uncond_embeddings_sequence: Sequence["torch.Tensor"],
        scorer: Any,
        prompt_for_loss: str,
        gd_steps: int = 1,
        gd_only_first_k_steps: Optional[int] = None,
        zt_optimizing: bool = True,
        zt_lr: float = 0.1,
        null_text_emb_optimizing: bool = False,
        null_text_emb_lr: float = 1e-3,
        normalize_grad: bool = True,
        clamp_grad_value: Optional[float] = 1.0,
        debug_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple["torch.Tensor", List["torch.Tensor"], List[Dict[str, Any]]]:
        if gd_steps < 0:
            raise ValueError("gd_steps must be >= 0")

        latents = start_latent.to(self.device, dtype=self.weights_dtype)
        cond_embeddings = cond_embeddings.to(self.device, dtype=self.weights_dtype)
        uncond_sequence = [u.to(self.device, dtype=self.weights_dtype) for u in uncond_embeddings_sequence]
        step_records: List[Dict[str, Any]] = []
        if gd_only_first_k_steps is None:
            gd_limit = len(self.scheduler.timesteps)
        else:
            gd_limit = max(0, min(int(gd_only_first_k_steps), len(self.scheduler.timesteps)))

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            timestep_int = int(timestep.item())
            uncond_step = uncond_sequence[step_index].detach().clone()
            do_opt = step_index < gd_limit and gd_steps > 0 and (zt_optimizing or null_text_emb_optimizing)
            gd_records: List[Dict[str, Any]] = []
            x0_img_before_guidance: Optional[torch.Tensor] = None

            if do_opt:
                z_var = latents.detach().clone().requires_grad_(zt_optimizing)
                uncond_var = uncond_step.detach().clone().requires_grad_(null_text_emb_optimizing)

                for gd_iter in range(gd_steps):
                    z_var = z_var.detach().clone().requires_grad_(zt_optimizing)
                    uncond_var = uncond_var.detach().clone().requires_grad_(null_text_emb_optimizing)

                    eps_pred = self.predict_noise_cfg(z_var, timestep, uncond_var, cond_embeddings)
                    x0_pred = self.predict_x0_from_eps(z_var, eps_pred, timestep_int)
                    x0_img = self.decode_latents_differentiable(x0_pred)
                    if x0_img_before_guidance is None:
                        x0_img_before_guidance = x0_img.detach().float().cpu()
                    if debug_hook is not None:
                        debug_hook(
                            {
                                "event": "gd_iter_before_update",
                                "denoise_step": step_index,
                                "timestep": timestep_int,
                                "gd_iter": gd_iter + 1,
                                "x_t_img_before": self.decode_latents(z_var.detach()),
                                "x0_img_before": x0_img.detach().float().cpu(),
                            }
                        )
                    scorer_output = scorer.score(image=x0_img, prompt=prompt_for_loss)
                    loss = scorer_output.loss.reshape(-1)[0]
                    score = scorer_output.score.reshape(-1)[0]
                    if not loss.requires_grad:
                        raise RuntimeError("Scorer loss is detached from graph. Guidance requires a differentiable loss.")

                    grad_targets: List[torch.Tensor] = []
                    target_names: List[str] = []
                    if zt_optimizing:
                        grad_targets.append(z_var)
                        target_names.append("z_t")
                    if null_text_emb_optimizing:
                        grad_targets.append(uncond_var)
                        target_names.append("null_text_emb")

                    grads = torch.autograd.grad(loss, grad_targets, retain_graph=False, create_graph=False, allow_unused=True)
                    grad_by_name = {name: grad for name, grad in zip(target_names, grads)}

                    with torch.no_grad():
                        z_updated = z_var.detach().clone()
                        uncond_updated = uncond_var.detach().clone()
                        x_t_before = z_updated.detach().clone()
                        record: Dict[str, Any] = {
                            "gd_iter": gd_iter + 1,
                            "score_before_update": float(score.detach().item()),
                            "loss_before_update": float(loss.detach().item()),
                            "zt_update_applied": False,
                            "null_text_emb_update_applied": False,
                        }

                        if zt_optimizing and grad_by_name.get("z_t") is not None:
                            grad_z = grad_by_name["z_t"]
                            grad_z, grad_z_norm_raw, grad_z_norm_processed, grad_z_ok = self._prepare_grad_for_update(
                                grad_z,
                                normalize_grad=normalize_grad,
                                clamp_grad_value=clamp_grad_value,
                            )
                            if grad_z_ok:
                                z_updated = z_updated.float() - float(zt_lr) * grad_z.float()
                                z_updated = z_updated.to(dtype=z_var.dtype)
                            record.update(
                                {
                                    "zt_grad_norm_raw": float(grad_z_norm_raw),
                                    "zt_grad_norm_processed": float(grad_z_norm_processed),
                                    "zt_update_applied": bool(grad_z_ok),
                                }
                            )

                        if null_text_emb_optimizing and grad_by_name.get("null_text_emb") is not None:
                            grad_e = grad_by_name["null_text_emb"]
                            grad_e, grad_e_norm_raw, grad_e_norm_processed, grad_e_ok = self._prepare_grad_for_update(
                                grad_e,
                                normalize_grad=normalize_grad,
                                clamp_grad_value=clamp_grad_value,
                            )
                            if grad_e_ok:
                                uncond_updated = uncond_updated.float() - float(null_text_emb_lr) * grad_e.float()
                                uncond_updated = uncond_updated.to(dtype=uncond_var.dtype)
                            record.update(
                                {
                                    "null_text_emb_grad_norm_raw": float(grad_e_norm_raw),
                                    "null_text_emb_grad_norm_processed": float(grad_e_norm_processed),
                                    "null_text_emb_update_applied": bool(grad_e_ok),
                                }
                            )

                        z_var = z_updated
                        uncond_var = uncond_updated
                        gd_records.append(record)
                        if debug_hook is not None:
                            debug_hook(
                                {
                                    "event": "gd_iter_after_update",
                                    "denoise_step": step_index,
                                    "timestep": timestep_int,
                                    "gd_iter": gd_iter + 1,
                                    "x_t_img_before": self.decode_latents(x_t_before),
                                    "x_t_img_after": self.decode_latents(z_var.detach()),
                                }
                            )

                latents = z_var.detach().clone()
                uncond_step = uncond_var.detach().clone()

            if debug_hook is not None:
                debug_hook(
                    {
                        "event": "step_done_gd",
                        "denoise_step": step_index,
                        "timestep": timestep_int,
                        "x_t_img_after_gd": self.decode_latents(latents.detach()),
                    }
                )

            with torch.no_grad():
                eps_pred = self.predict_noise_cfg(latents, timestep, uncond_step, cond_embeddings)
                x0_pred = self.predict_x0_from_eps(latents, eps_pred, timestep_int)
                x0_img = self.decode_latents_differentiable(x0_pred)
                eval_output = scorer.score(image=x0_img, prompt=prompt_for_loss)
                score_after = float(eval_output.score.reshape(-1)[0].detach().item())
                loss_after = float(eval_output.loss.reshape(-1)[0].detach().item())
                if debug_hook is not None:
                    debug_hook(
                        {
                            "event": "step_before_denoise",
                            "denoise_step": step_index,
                            "timestep": timestep_int,
                            "x0_img_before_denoise": x0_img.detach().float().cpu(),
                            "x0_img_before_guidance": x0_img_before_guidance,
                        }
                    )
                latents = self.prev_step(eps_pred, timestep_int, latents)

            uncond_sequence[step_index] = uncond_step.detach().clone()
            step_records.append(
                {
                    "denoise_step": step_index,
                    "timestep": timestep_int,
                    "gd_applied": bool(do_opt),
                    "score_after_guidance": score_after,
                    "loss_after_guidance": loss_after,
                    "gd_stats": gd_records,
                }
            )

        latents_cpu = latents.float().cpu()
        uncond_sequence_cpu = [u.float().cpu() for u in uncond_sequence]
        return latents_cpu, uncond_sequence_cpu, step_records

    def run(
        self,
        image_bchw: "torch.Tensor",
        source_prompt: str,
        edit_prompt: Optional[str],
        num_null_text_steps: int,
        null_text_lr: float,
        early_stop_eps: float,
    ) -> InversionArtifacts:
        image_bchw = image_bchw.float().cpu()
        image_latent = self.encode_image(image_bchw)
        base_uncond_embeddings = self.encode_text("")
        cond_embeddings = self.encode_text(source_prompt)
        ddim_latents = self.ddim_inversion(image_latent, cond_embeddings)
        null_text_embeddings, loss_history = self.null_text_optimization(
            ddim_latents=ddim_latents,
            cond_embeddings=cond_embeddings,
            base_uncond_embeddings=base_uncond_embeddings,
            num_inner_steps=num_null_text_steps,
            lr=null_text_lr,
            early_stop_eps=early_stop_eps,
        )

        start_latent = ddim_latents[-1]
        reconstruction_latent = self.sample_with_uncond_sequence(start_latent, cond_embeddings, null_text_embeddings)
        reconstruction = self.decode_latents(reconstruction_latent)

        edited = None
        if edit_prompt is not None:
            edit_embeddings = self.encode_text(edit_prompt)
            edited_latent = self.sample_with_uncond_sequence(start_latent, edit_embeddings, null_text_embeddings)
            edited = self.decode_latents(edited_latent)

        return InversionArtifacts(
            source_prompt=source_prompt,
            edit_prompt=edit_prompt,
            image_latent=image_latent.float().cpu(),
            ddim_latents=list(ddim_latents),
            null_text_embeddings=list(null_text_embeddings),
            null_text_loss_history=list(loss_history),
            reconstruction=reconstruction,
            edited=edited,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run null-text inversion for a single image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source-prompt", type=str, required=True)
    parser.add_argument("--edit-prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights-dtype", type=str, default="float32")
    parser.add_argument("--num-ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-null-text-steps", type=int, default=10)
    parser.add_argument("--null-text-lr", type=float, default=1e-2)
    parser.add_argument("--early-stop-eps", type=float, default=1e-5)
    return parser


def save_artifacts(
    output_dir: Path,
    image_path: Path,
    artifacts: InversionArtifacts,
    resized_input: "torch.Tensor",
) -> None:
    require_ml_deps()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_image = output_dir / image_path.name
    if not target_image.exists():
        Image.open(image_path).convert("RGB").save(target_image)

    from vlm_guidance_project.vlm_guidance.utils.images import save_diff_image, save_image_tensor

    save_image_tensor(resized_input, output_dir / "input_resized.png")
    save_image_tensor(artifacts.reconstruction, output_dir / "reconstruction.png")
    save_diff_image(resized_input, artifacts.reconstruction, output_dir / "reconstruction_abs_diff.png")
    if artifacts.edited is not None and artifacts.edit_prompt is not None:
        slug = artifacts.edit_prompt.lower().replace(" ", "_")[:40]
        save_image_tensor(artifacts.edited, output_dir / f"edited_{slug}.png")

    torch.save(artifacts.image_latent, output_dir / "image_latent.pt")
    torch.save(artifacts.ddim_latents, output_dir / "ddim_latents.pt")
    torch.save(artifacts.null_text_embeddings, output_dir / "null_text_embeddings.pt")

    metadata = {
        "image_path": str(image_path),
        "source_prompt": artifacts.source_prompt,
        "edit_prompt": artifacts.edit_prompt,
        "num_ddim_latents": len(artifacts.ddim_latents),
        "num_null_text_embeddings": len(artifacts.null_text_embeddings),
        "null_text_loss_history": artifacts.null_text_loss_history,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    resized_input = load_image(args.image, size=(512, 512))
    pipeline = NullTextInversionPipeline(
        model_id=args.model_id,
        device=args.device,
        weights_dtype=args.weights_dtype,
        num_ddim_steps=args.num_ddim_steps,
        guidance_scale=args.guidance_scale,
    )
    artifacts = pipeline.run(
        image_bchw=resized_input,
        source_prompt=args.source_prompt,
        edit_prompt=args.edit_prompt,
        num_null_text_steps=args.num_null_text_steps,
        null_text_lr=args.null_text_lr,
        early_stop_eps=args.early_stop_eps,
    )
    save_artifacts(args.output_dir, args.image, artifacts, resized_input)


if __name__ == "__main__":
    main()
