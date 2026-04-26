from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from vlm_guidance_project.vlm_guidance.diffusion.base import BaseDiffusionBackend
from vlm_guidance_project.vlm_guidance.generation.base import BaseText2ImagePipeline
from vlm_guidance_project.vlm_guidance.scorers.base import BaseDifferentiableScorer
from vlm_guidance_project.vlm_guidance.utils.images import save_image_tensor
from vlm_guidance_project.vlm_guidance.utils.io import save_json
from vlm_guidance_project.vlm_guidance.utils.tensorboard import TensorBoardRunLogger


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


class VanillaSD15WithVLMRunner:
    def __init__(
        self,
        diffusion: BaseDiffusionBackend,
        scorer: BaseDifferentiableScorer,
    ) -> None:
        self.diffusion = diffusion
        self.scorer = scorer
        self.verbose_vlm = False
        self.yes_no_loss = True
        self.vlm_num_tokens = 16
        self.verbose_vlm_prompt_template = 'Does this figure show "{}"? And provide explanation'
        self.vqa_vlm_prompt_template = 'Does this figure show "{}"? Please answer yes or no'
        self.vqa_question_template = self.vqa_vlm_prompt_template
        self.verbose_question_template = self.verbose_vlm_prompt_template

    def _render_question(self, question_template: str | None, prompt: str) -> str | None:
        if question_template is None:
            return None
        return question_template.format(prompt)

    def configure_verbose_vlm(
        self,
        verbose_vlm: bool = False,
        yes_no_loss: bool = True,
        vlm_num_tokens: int = 16,
        verbose_vlm_prompt_template: str = 'Does this figure show "{}"? And provide explanation',
        vqa_vlm_prompt_template: str = 'Does this figure show "{}"? Please answer yes or no',
    ) -> None:
        self.verbose_vlm = verbose_vlm
        self.yes_no_loss = yes_no_loss
        self.vlm_num_tokens = vlm_num_tokens
        self.verbose_vlm_prompt_template = verbose_vlm_prompt_template
        self.vqa_vlm_prompt_template = vqa_vlm_prompt_template
        self.vqa_question_template = self.vqa_vlm_prompt_template
        self.verbose_question_template = self.verbose_vlm_prompt_template

    def _generate_verbose_answer(self, image: torch.Tensor, prompt: str) -> str | None:
        if not self.verbose_vlm or not hasattr(self.scorer, "generate_answer"):
            return None
        answers = self.scorer.generate_answer(
            image=image.detach(),
            prompt=prompt,
            question_template=self.verbose_question_template,
            max_new_tokens=self.vlm_num_tokens,
        )
        if isinstance(answers, list):
            return answers[0] if answers else ""
        return str(answers)

    def run(
        self,
        run_dir: str | Path,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
        image_filename_template: str = "img_{index:02d}.png",
        prompt_filename: str = "prompt.txt",
        summary_filename: str = "result_summary.json",
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / prompt_filename).write_text(prompt, encoding="utf-8")
        x0_dir = run_dir / "x0_vanilla"
        x0_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorBoardRunLogger(run_dir / "tensorboard")
        try:
            with torch.no_grad():
                vqa_vlm_prompt = self._render_question(self.vqa_question_template, prompt)
                verbose_vlm_prompt = self._render_question(self.verbose_question_template, prompt) if self.verbose_vlm else None
                if vqa_vlm_prompt is not None:
                    tb_logger.add_text("vanilla_sd15/prompts/vqa_vlm", vqa_vlm_prompt, global_step=0)
                if verbose_vlm_prompt is not None:
                    tb_logger.add_text("vanilla_sd15/prompts/verbose_vlm", verbose_vlm_prompt, global_step=0)
                text_embeds = self.diffusion.encode_prompt(prompt, negative_prompt)
                self.diffusion.set_timesteps(num_inference_steps)
                latents = self.diffusion.init_latents(
                    height=height,
                    width=width,
                    batch_size=batch_size,
                    seed=seed,
                )

                step_records: List[Dict[str, Any]] = []
                for i, t in enumerate(self.diffusion.timesteps):
                    eps_pred = self.diffusion.predict_eps_with_cfg(
                        x_t=latents,
                        t=t,
                        text_embeds=text_embeds,
                        guidance_scale=guidance_scale,
                    )
                    x0_pred = self.diffusion.predict_x0_from_eps(
                        x_t=latents,
                        eps_pred=eps_pred,
                        t=t,
                    )
                    x0_img = self.diffusion.decode_latents(x0_pred)
                    save_image_tensor(x0_img, x0_dir / f"t_{i:04d}_before_denoise.png")
                    tb_logger.add_image(
                        tag="vanilla_sd15/images/x0/before_denoise",
                        image=x0_img,
                        global_step=i,
                    )
                    verbose_vlm_answer = self._generate_verbose_answer(x0_img, prompt)
                    if verbose_vlm_answer is not None:
                        tb_logger.add_text(
                            tag="vanilla_sd15/verbose_vlm/before_denoise",
                            text=verbose_vlm_answer,
                            global_step=i,
                        )

                    scorer_output = self.scorer.score(
                        image=x0_img,
                        prompt=prompt,
                        question_template=self.vqa_question_template,
                        yes_no_loss=self.yes_no_loss,
                    )
                    vlm_loss = scorer_output.loss.reshape(-1).detach().cpu().tolist()
                    vlm_score = scorer_output.score.reshape(-1).detach().cpu().tolist()
                    vlm_loss_type = str(scorer_output.extras.get("loss_type", "unknown"))

                    step_records.append({
                        "timestep": int(t.item()),
                        "denoise_step": i,
                        "vlm_loss": vlm_loss,
                        "vlm_score": vlm_score,
                        "vlm_loss_type": vlm_loss_type,
                        "verbose_vlm_answer": verbose_vlm_answer,
                    })

                    latents = self.diffusion.scheduler_step(eps_pred, t, latents)

                final_images_tensor = self.diffusion.decode_latents(latents)

            image_paths: List[str] = []
            for idx in range(final_images_tensor.shape[0]):
                image = final_images_tensor[idx].detach().cpu().clamp(0, 1)
                image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
                path = run_dir / image_filename_template.format(index=idx)
                Image.fromarray(image).save(path)
                image_paths.append(str(path))

            result = {
                "pipeline_name": "vanilla_sd15",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "batch_size": batch_size,
                "image_paths": image_paths,
                "vlm_loss_type": "yes_no_margin" if self.yes_no_loss else "yes_cross_entropy",
                "vqa_vlm_question_template": self.vqa_question_template,
                "verbose_vlm_question_template": self.verbose_question_template,
                "vqa_vlm_prompt": vqa_vlm_prompt,
                "verbose_vlm_prompt": verbose_vlm_prompt,
                "steps": step_records,
            }
            save_json(result, run_dir / summary_filename)
            tb_logger.add_image("vanilla_sd15/images/final", final_images_tensor, global_step=len(step_records))
            tb_logger.log_summary_scalars(result, tag_prefix="vanilla_sd15")
            return result
        finally:
            tb_logger.close()
