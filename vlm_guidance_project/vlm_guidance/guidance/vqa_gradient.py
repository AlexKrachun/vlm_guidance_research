from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vlm_guidance_project.vlm_guidance.diffusion.base import BaseDiffusionBackend
from vlm_guidance_project.vlm_guidance.scorers.base import BaseDifferentiableScorer
from vlm_guidance_project.vlm_guidance.utils.debug import tensor_stats
from vlm_guidance_project.vlm_guidance.utils.images import save_diff_image, save_image_tensor
from vlm_guidance_project.vlm_guidance.utils.io import save_json
from vlm_guidance_project.vlm_guidance.utils.tensorboard import TensorBoardRunLogger


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
    vanilla_calc_vlm_loss: bool = False
    flux1: bool = False
    verbose_vlm: bool = False
    yes_no_loss: bool = True
    vlm_num_tokens: int = 16
    verbose_vlm_prompt_template: str = 'Does this figure show "{}"? And provide explanation'
    vqa_vlm_prompt_template: str = 'Does this figure show "{}"? Please answer yes or no'


@dataclass
class VQAGuidanceConfig:
    gd_steps: int = 2
    gd_lr: float = 0.1
    gd_only_first_k_steps: int = 10
    normalize_grad: bool = True
    clamp_grad_value: Optional[float] = 1.0
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
        self.vqa_question_template = self.run_cfg.vqa_vlm_prompt_template
        self.verbose_question_template = self.run_cfg.verbose_vlm_prompt_template

    def _render_question(self, question_template: str | None) -> str | None:
        if question_template is None or self.run_cfg.prompt is None:
            return None
        return question_template.format(self.run_cfg.prompt)

    def _generate_verbose_answer(self, image: torch.Tensor) -> str | None:
        if not self.run_cfg.verbose_vlm or not hasattr(self.scorer, "generate_answer"):
            return None
        answers = self.scorer.generate_answer(
            image=image.detach(),
            prompt=self.run_cfg.prompt,
            question_template=self.verbose_question_template,
            max_new_tokens=self.run_cfg.vlm_num_tokens,
        )
        if isinstance(answers, list):
            return answers[0] if answers else ""
        return str(answers)

    def _prepare_grad(self, grad: torch.Tensor) -> tuple[torch.Tensor, float, float, bool]:
        grad_fp32 = grad.detach().float()
        raw_finite = torch.isfinite(grad_fp32).all()
        raw_norm = torch.linalg.vector_norm(grad_fp32).detach()

        if self.guidance_cfg.normalize_grad and raw_finite and torch.isfinite(raw_norm) and raw_norm.item() > 0:
            grad_fp32 = grad_fp32 / raw_norm

        if self.guidance_cfg.clamp_grad_value is not None:
            grad_fp32 = grad_fp32.clamp(-self.guidance_cfg.clamp_grad_value, self.guidance_cfg.clamp_grad_value)

        processed_finite = torch.isfinite(grad_fp32).all()
        processed_norm = torch.linalg.vector_norm(grad_fp32).detach()
        is_valid = bool(raw_finite.item() and processed_finite.item() and torch.isfinite(raw_norm).item() and torch.isfinite(processed_norm).item())
        return grad_fp32.to(dtype=grad.dtype), float(raw_norm.item()), float(processed_norm.item()), is_valid

    def _make_artifact_dirs(self, run_dir: Path) -> Dict[str, Optional[Path]]:
        if not self.guidance_cfg.save_debug_tensors:
            return {"xt": None, "x0": None, "xdiff": None, "x0diff": None, "x0diff_denoise": None}
        dirs = {
            "xt": run_dir / "xt_gd",
            "x0": run_dir / "x0_gd",
            "xdiff": run_dir / "xdiff_gd",
            "x0diff": run_dir / "x0diff_gd",
            "x0diff_denoise": run_dir / "x0diff_denoise",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def run(
        self,
        run_dir: str | Path = ".",
        prompt_filename: str = "prompt.txt",
        summary_filename: str = "result_summary.json",
    ) -> Dict[str, Any]:
        if not self.run_cfg.prompt:
            raise ValueError("run.prompt must be provided for VQA guidance runs.")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._make_artifact_dirs(run_dir)
        tb_logger = TensorBoardRunLogger(run_dir / "tensorboard")
        try:
            (run_dir / prompt_filename).write_text(self.run_cfg.prompt, encoding="utf-8")
            vqa_vlm_prompt = self._render_question(self.vqa_question_template)
            verbose_vlm_prompt = self._render_question(self.verbose_question_template) if self.run_cfg.verbose_vlm else None
            if vqa_vlm_prompt is not None:
                tb_logger.add_text("vqa_score/prompts/vqa_vlm", vqa_vlm_prompt, global_step=0)
            if verbose_vlm_prompt is not None:
                tb_logger.add_text("vqa_score/prompts/verbose_vlm", verbose_vlm_prompt, global_step=0)
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
                x0_img_before_guidance: Optional[torch.Tensor] = None

                if do_gd:
                    for gd_iter in range(self.guidance_cfg.gd_steps):
                        x_t_gd = x_t_gd.detach().clone().requires_grad_(True)
                        if artifacts["xt"] is not None:
                            xt_img_before = self.diffusion.decode_latents(x_t_gd)
                            save_image_tensor(xt_img_before, artifacts["xt"] / f"{filename_base}_gd_{gd_iter + 1:02d}_before_gd_step.png")
                            tb_logger.add_image(
                                tag=f"vqa_score/images/xt/gd_iter_{gd_iter + 1:02d}/before_gd_step",
                                image=xt_img_before,
                                global_step=i,
                            )

                        eps_pred = self.diffusion.predict_eps_with_cfg(x_t=x_t_gd, t=t, text_embeds=text_embeds, guidance_scale=self.run_cfg.guidance_scale)
                        x0_pred = self.diffusion.predict_x0_from_eps(x_t=x_t_gd, eps_pred=eps_pred, t=t)
                        x0_img = self.diffusion.decode_latents(x0_pred)
                        if x0_img_before_guidance is None:
                            x0_img_before_guidance = x0_img.detach().clone()

                        if artifacts["x0"] is not None:
                            save_image_tensor(x0_img, artifacts["x0"] / f"{filename_base}_gd_{gd_iter + 1:02d}_before_gd_step.png")
                            tb_logger.add_image(
                                tag=f"vqa_score/images/x0/gd_iter_{gd_iter + 1:02d}/before_gd_step",
                                image=x0_img,
                                global_step=i,
                            )

                        verbose_vlm_answer = self._generate_verbose_answer(x0_img)
                        if verbose_vlm_answer is not None:
                            tb_logger.add_text(
                                tag=f"vqa_score/verbose_vlm/gd_iter_{gd_iter + 1:02d}",
                                text=verbose_vlm_answer,
                                global_step=i,
                            )

                        scorer_output = self.scorer.score(
                            image=x0_img,
                            prompt=self.run_cfg.prompt,
                            question_template=self.vqa_question_template,
                            yes_no_loss=self.run_cfg.yes_no_loss,
                        )
                        loss = scorer_output.loss.reshape(-1)[0]
                        score = scorer_output.score.reshape(-1)[0]
                        loss_type = str(scorer_output.extras.get("loss_type", "unknown"))
                        if not loss.requires_grad:
                            raise RuntimeError("Scorer loss is detached from graph. Guidance requires a differentiable loss.")

                        grad = torch.autograd.grad(loss, x_t_gd, retain_graph=False, create_graph=False)[0]

                        with torch.no_grad():
                            xt_before = x_t_gd.detach().clone()
                            raw_grad = grad.detach().clone()
                            grad, grad_norm_raw, grad_norm_processed, grad_ok = self._prepare_grad(grad)
                            if grad_ok:
                                updated = x_t_gd.detach().float() - self.guidance_cfg.gd_lr * grad.float()
                                x_t_gd = updated.to(dtype=x_t_gd.dtype)
                            xt_after = x_t_gd.detach().clone()

                            if artifacts["xdiff"] is not None:
                                xt_diff_before = self.diffusion.decode_latents(xt_before)
                                xt_diff_after = self.diffusion.decode_latents(xt_after)
                                save_diff_image(
                                    xt_diff_before,
                                    xt_diff_after,
                                    artifacts["xdiff"] / f"{filename_base}_gd_{gd_iter + 1:02d}.png",
                                )
                                tb_logger.add_image(
                                    tag=f"vqa_score/images/xdiff/gd_iter_{gd_iter + 1:02d}",
                                    image=(xt_diff_after - xt_diff_before).abs().mul(2).div((xt_diff_after - xt_diff_before).abs().mul(2).max().clamp(min=1e-8, max=1)),
                                    global_step=i,
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
                                    "grad_raw": tensor_stats("grad_raw", raw_grad),
                                    "grad_processed": tensor_stats("grad_processed", grad),
                                }
                            })

                        gd_records.append({
                            "timestep": int(t.item()),
                            "denoise_step": i,
                            "gd_iter": gd_iter + 1,
                            "grad_norm_raw": float(grad_norm_raw),
                            "grad_norm_processed": float(grad_norm_processed),
                            "score_before_update": float(score.detach().item()),
                            "loss_before_update": float(loss.detach().item()),
                            "vlm_loss_type": loss_type,
                            "grad_update_applied": bool(grad_ok),
                            "verbose_vlm_answer": verbose_vlm_answer,
                        })
                else:
                    x_t_gd = x_t_orig.detach().clone()

                if artifacts["xt"] is not None:
                    xt_done_gd = self.diffusion.decode_latents(x_t_gd)
                    save_image_tensor(xt_done_gd, artifacts["xt"] / f"{filename_base}_done_gd_step.png")
                    tb_logger.add_image(
                        tag="vqa_score/images/xt/done_gd_step",
                        image=xt_done_gd,
                        global_step=i,
                    )

                with torch.no_grad():
                    eps_pred = self.diffusion.predict_eps_with_cfg(x_t=x_t_gd, t=t, text_embeds=text_embeds, guidance_scale=self.run_cfg.guidance_scale)
                    x0 = self.diffusion.predict_x0_from_eps(x_t_gd, eps_pred, t)
                    x0_img = self.diffusion.decode_latents(x0)
                    if artifacts["x0"] is not None:
                        save_image_tensor(x0_img, artifacts["x0"] / f"{filename_base}_before_denoise.png")
                        tb_logger.add_image(
                            tag="vqa_score/images/x0/before_denoise",
                            image=x0_img,
                            global_step=i,
                        )
                    if artifacts["x0diff"] is not None and x0_img_before_guidance is not None:
                        x0diff_cumulative = (x0_img - x0_img_before_guidance).abs() * 2
                        x0diff_cumulative = x0diff_cumulative / x0diff_cumulative.max().clamp(min=1e-8, max=1)
                        save_diff_image(
                            x0_img_before_guidance,
                            x0_img,
                            artifacts["x0diff"] / f"{filename_base}_cumulative_before_denoise.png",
                        )
                        tb_logger.add_image(
                            tag="vqa_score/images/x0diff/cumulative_before_denoise",
                            image=x0diff_cumulative,
                            global_step=i,
                        )
                    eval_output = self.scorer.score(
                        image=x0_img,
                        prompt=self.run_cfg.prompt,
                        question_template=self.vqa_question_template,
                        yes_no_loss=self.run_cfg.yes_no_loss,
                    )
                    vqa_after_gd = float(eval_output.score.reshape(-1)[0].item())
                    latents = self.diffusion.scheduler_step(eps_pred, t, x_t_gd)
                    denoise_step_norm = (latents - x_t_gd).norm().detach().item()
                    if artifacts["x0diff_denoise"] is not None and i + 1 < len(self.diffusion.timesteps):
                        next_t = self.diffusion.timesteps[i + 1]
                        eps_pred_after_denoise = self.diffusion.predict_eps_with_cfg(
                            x_t=latents,
                            t=next_t,
                            text_embeds=text_embeds,
                            guidance_scale=self.run_cfg.guidance_scale,
                        )
                        x0_after_denoise = self.diffusion.predict_x0_from_eps(
                            x_t=latents,
                            eps_pred=eps_pred_after_denoise,
                            t=next_t,
                        )
                        x0_img_after_denoise = self.diffusion.decode_latents(x0_after_denoise)
                        x0diff_denoise = (x0_img_after_denoise - x0_img).abs() * 2
                        x0diff_denoise = x0diff_denoise / x0diff_denoise.max().clamp(min=1e-8, max=1)
                        save_diff_image(
                            x0_img,
                            x0_img_after_denoise,
                            artifacts["x0diff_denoise"] / f"{filename_base}_after_denoise.png",
                        )
                        tb_logger.add_image(
                            tag="vqa_score/images/x0diff_denoise/after_denoise",
                            image=x0diff_denoise,
                            global_step=i,
                        )

                step_records.append({
                    "timestep": int(t.item()),
                    "denoise_step": i,
                    "gd_applied": bool(do_gd),
                    "vqa_score_after_gd": vqa_after_gd,
                    "denoise_step_norm": float(denoise_step_norm),
                    "gd_stats": gd_records if do_gd else [],
                })

            with torch.no_grad():
                final_images = self.diffusion.decode_latents(latents)
                final_output = self.scorer.score(
                    image=final_images,
                    prompt=self.run_cfg.prompt,
                    question_template=self.vqa_question_template,
                    yes_no_loss=self.run_cfg.yes_no_loss,
                )
                final_score = float(final_output.score.reshape(-1)[0].item())

            final_image_path = run_dir / self.guidance_cfg.final_img_filename
            save_image_tensor(final_images, final_image_path)

            meta = {
                "pipeline_name": "vqa_score",
                "run": asdict(self.run_cfg),
                "guidance": asdict(self.guidance_cfg),
                "vlm_loss_type": "yes_no_margin" if self.run_cfg.yes_no_loss else "yes_cross_entropy",
                "final_image_path": str(final_image_path),
                "final_score": final_score,
                "vqa_vlm_question_template": self.vqa_question_template,
                "verbose_vlm_question_template": self.verbose_question_template,
                "vqa_vlm_prompt": vqa_vlm_prompt,
                "verbose_vlm_prompt": verbose_vlm_prompt,
                "steps": step_records,
            }
            save_json(meta, run_dir / summary_filename)
            tb_logger.add_image("vqa_score/images/final", final_images, global_step=len(step_records))
            tb_logger.log_summary_scalars(meta, tag_prefix="vqa_score")
            return meta
        finally:
            tb_logger.close()
