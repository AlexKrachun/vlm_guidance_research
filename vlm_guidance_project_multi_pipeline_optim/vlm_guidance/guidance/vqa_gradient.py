from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vlm_guidance.diffusion.base import BaseDiffusionBackend
from vlm_guidance.scorers.base import BaseDifferentiableScorer
from vlm_guidance.utils.debug import tensor_stats
from vlm_guidance.utils.images import save_diff_image, save_image_tensor
from vlm_guidance.utils.io import save_json


@dataclass
class RunConfig:
    prompt: Optional[str] = None
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = 42
    batch_size: int = 1
    vqa_score: bool = True
    vanilla_sd: bool = False
    flux1: bool = False


@dataclass
class VQAGuidanceConfig:
    gd_steps: int = 2
    gd_lr: float = 0.1
    gd_only_first_k_steps: int = 10
    normalize_grad: bool = True
    clamp_grad_value: Optional[float] = 1.0
    save_only_final_img: bool = True
    final_img_filename: str = "img.png"
    save_debug_tensors: bool = False


class VQAGradientGuidanceRunner:
    def __init__(
        self,
        diffusion: BaseDiffusionBackend,
        scorer: BaseDifferentiableScorer,
        run_cfg: RunConfig,
        guidance_cfg: VQAGuidanceConfig,
    ) -> None:
        self.diffusion = diffusion
        self.scorer = scorer
        self.run_cfg = run_cfg
        self.guidance_cfg = guidance_cfg

    def _make_artifact_dirs(self, run_dir: Path) -> Dict[str, Optional[Path]]:
        if self.guidance_cfg.save_only_final_img:
            return {"xt": None, "x0": None, "xdiff": None}
        dirs = {
            "xt": run_dir / "xt_gd",
            "x0": run_dir / "x0_gd",
            "xdiff": run_dir / "xdiff_gd",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def run(
        self,
        run_dir: str | Path = ".",
        prompt_filename: str = "prompt.txt",
        scores_filename: str = "scores.json",
        summary_filename: str = "result_summary.json",
    ) -> Dict[str, Any]:
        if not self.run_cfg.prompt:
            raise ValueError("run.prompt must be provided for VQA guidance runs.")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._make_artifact_dirs(run_dir)

        (run_dir / prompt_filename).write_text(self.run_cfg.prompt, encoding="utf-8")
        text_embeds = self.diffusion.encode_prompt(self.run_cfg.prompt, self.run_cfg.negative_prompt)
        self.diffusion.set_timesteps(self.run_cfg.num_inference_steps)
        latents = self.diffusion.init_latents(
            height=self.run_cfg.height,
            width=self.run_cfg.width,
            batch_size=self.run_cfg.batch_size,
            seed=self.run_cfg.seed,
        )

        n_digits = max(4, len(str(len(self.diffusion.timesteps))))
        step_records: List[Dict[str, Any]] = []
        gd_limit = max(0, min(int(self.guidance_cfg.gd_only_first_k_steps), len(self.diffusion.timesteps)))

        for i, t in enumerate(self.diffusion.timesteps):
            filename_base = f"t_{i:0{n_digits}d}"
            x_t_orig = latents.detach().clone()
            x_t_gd = x_t_orig.detach().clone().requires_grad_(True)
            do_gd = i < gd_limit and self.guidance_cfg.gd_steps > 0
            gd_records: List[Dict[str, Any]] = []

            if do_gd:
                for gd_iter in range(self.guidance_cfg.gd_steps):
                    x_t_gd = x_t_gd.detach().clone().requires_grad_(True)

                    if artifacts["xt"] is not None:
                        xt_img_before = self.diffusion.decode_latents(x_t_gd)
                        save_image_tensor(xt_img_before, artifacts["xt"] / f"{filename_base}_gd_{gd_iter + 1:02d}_before_gd_step.png")

                    eps_pred = self.diffusion.predict_eps_with_cfg(x_t=x_t_gd, t=t, text_embeds=text_embeds, guidance_scale=self.run_cfg.guidance_scale)
                    x0_pred = self.diffusion.predict_x0_from_eps(x_t=x_t_gd, eps_pred=eps_pred, t=t)
                    x0_img = self.diffusion.decode_latents(x0_pred)

                    if artifacts["x0"] is not None:
                        save_image_tensor(x0_img, artifacts["x0"] / f"{filename_base}_gd_{gd_iter + 1:02d}_before_gd_step.png")

                    scorer_output = self.scorer.score(image=x0_img, prompt=self.run_cfg.prompt)
                    loss = scorer_output.loss.reshape(-1)[0]
                    score = scorer_output.score.reshape(-1)[0]
                    if not loss.requires_grad:
                        raise RuntimeError("Scorer loss is detached from graph. Guidance requires a differentiable loss.")

                    grad = torch.autograd.grad(loss, x_t_gd, retain_graph=False, create_graph=False)[0]

                    with torch.no_grad():
                        xt_before = x_t_gd.detach().clone()
                        grad_norm_raw = grad.norm().detach().item()
                        if self.guidance_cfg.normalize_grad:
                            grad = grad / (grad.norm() + 1e-8)
                        if self.guidance_cfg.clamp_grad_value is not None:
                            grad = grad.clamp(-self.guidance_cfg.clamp_grad_value, self.guidance_cfg.clamp_grad_value)
                        x_t_gd -= self.guidance_cfg.gd_lr * grad
                        xt_after = x_t_gd.detach().clone()

                        if artifacts["xdiff"] is not None:
                            save_diff_image(
                                self.diffusion.decode_latents(xt_before),
                                self.diffusion.decode_latents(xt_after),
                                artifacts["xdiff"] / f"{filename_base}_gd_{gd_iter + 1:02d}.png",
                            )

                    if self.guidance_cfg.save_debug_tensors:
                        gd_records.append({
                            "debug": {
                                "x_t_gd_after_update": tensor_stats("x_t_gd_after_update", x_t_gd),
                                "eps_pred": tensor_stats("eps_pred", eps_pred),
                                "x0_pred": tensor_stats("x0_pred", x0_pred),
                                "x0_img": tensor_stats("x0_img", x0_img),
                                "loss": tensor_stats("loss", loss),
                                "score": tensor_stats("score", score),
                                "grad": tensor_stats("grad", grad),
                            }
                        })

                    gd_records.append({
                        "timestep": int(t.item()),
                        "denoise_step": i,
                        "gd_iter": gd_iter + 1,
                        "grad_norm_raw": float(grad_norm_raw),
                        "score_before_update": float(score.detach().item()),
                        "loss_before_update": float(loss.detach().item()),
                    })
            else:
                x_t_gd = x_t_orig.detach().clone()

            if artifacts["xt"] is not None:
                save_image_tensor(self.diffusion.decode_latents(x_t_gd), artifacts["xt"] / f"{filename_base}_done_gd_step.png")

            with torch.no_grad():
                eps_pred = self.diffusion.predict_eps_with_cfg(x_t=x_t_gd, t=t, text_embeds=text_embeds, guidance_scale=self.run_cfg.guidance_scale)
                x0 = self.diffusion.predict_x0_from_eps(x_t_gd, eps_pred, t)
                x0_img = self.diffusion.decode_latents(x0)
                if artifacts["x0"] is not None:
                    save_image_tensor(x0_img, artifacts["x0"] / f"{filename_base}_before_denoise.png")
                eval_output = self.scorer.score(image=x0_img, prompt=self.run_cfg.prompt)
                vqa_after_gd = float(eval_output.score.reshape(-1)[0].item())
                latents = self.diffusion.scheduler_step(eps_pred, t, x_t_gd)

            step_records.append({
                "timestep": int(t.item()),
                "denoise_step": i,
                "gd_applied": bool(do_gd),
                "vqa_score_after_gd": vqa_after_gd,
                "gd_stats": gd_records if do_gd else [],
            })

        with torch.no_grad():
            final_images = self.diffusion.decode_latents(latents)
            final_output = self.scorer.score(image=final_images, prompt=self.run_cfg.prompt)
            final_score = float(final_output.score.reshape(-1)[0].item())

        final_image_path = run_dir / self.guidance_cfg.final_img_filename
        save_image_tensor(final_images, final_image_path)

        meta = {
            "pipeline_name": "vqa_score",
            "run": asdict(self.run_cfg),
            "guidance": asdict(self.guidance_cfg),
            "final_image_path": str(final_image_path),
            "final_score": final_score,
            "steps": step_records,
        }
        save_json(meta, run_dir / scores_filename)
        save_json(meta, run_dir / summary_filename)
        return meta
