from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from vlm_guidance.generation.base import Text2ImageRunner
from vlm_guidance.guidance.vqa_gradient import VQAGradientGuidanceRunner
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)


PIPELINE_ORDER: List[str] = ["vqa_score", "vanilla_sd", "flux1"]


def get_selected_pipelines(cfg: DictConfig) -> List[str]:
    selected: List[str] = []
    if cfg.run.vqa_score:
        selected.append("vqa_score")
    if cfg.run.vanilla_sd:
        selected.append("vanilla_sd")
    if cfg.run.flux1:
        selected.append("flux1")
    if not selected:
        raise ValueError("At least one pipeline must be enabled: run.vqa_score, run.vanilla_sd or run.flux1.")
    return selected


def build_common_kwargs(cfg: DictConfig, prompt: str) -> Dict[str, Any]:
    return dict(
        prompt=prompt,
        negative_prompt=cfg.run.negative_prompt,
        height=cfg.run.height,
        width=cfg.run.width,
        num_inference_steps=cfg.run.num_inference_steps,
        guidance_scale=cfg.run.guidance_scale,
        seed=cfg.run.seed,
        batch_size=cfg.run.batch_size,
    )


def instantiate_pipeline_runner(cfg: DictConfig, pipeline_name: str):
    if pipeline_name == "vqa_score":
        log.info("Instantiating VQA guidance pipeline")
        diffusion = instantiate(cfg.diffusion)
        scorer = instantiate(cfg.scorer)
        run_cfg = instantiate(cfg.run)
        guidance_cfg = instantiate(cfg.algorithm)
        return VQAGradientGuidanceRunner(
            diffusion=diffusion,
            scorer=scorer,
            run_cfg=run_cfg,
            guidance_cfg=guidance_cfg,
        )

    if pipeline_name == "vanilla_sd":
        log.info("Instantiating vanilla SD1.5 pipeline")
        vanilla_pipe = instantiate(cfg.vanilla_sd)
        return Text2ImageRunner(vanilla_pipe)

    if pipeline_name == "flux1":
        log.info("Instantiating FLUX.1-dev pipeline")
        flux_pipe = instantiate(cfg.flux1)
        return Text2ImageRunner(flux_pipe)

    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def run_single_pipeline(
    runner,
    cfg: DictConfig,
    pipeline_name: str,
    run_dir: Path,
    prompt: str,
    flatten_output: bool = False,
) -> Dict[str, Any]:
    common_kwargs = build_common_kwargs(cfg, prompt)

    if pipeline_name == "vqa_score":
        runner.run_cfg.prompt = prompt
        runner.run_cfg.negative_prompt = cfg.run.negative_prompt
        runner.run_cfg.height = cfg.run.height
        runner.run_cfg.width = cfg.run.width
        runner.run_cfg.num_inference_steps = cfg.run.num_inference_steps
        runner.run_cfg.guidance_scale = cfg.run.guidance_scale
        runner.run_cfg.seed = cfg.run.seed
        runner.run_cfg.batch_size = cfg.run.batch_size
        if flatten_output:
            runner.guidance_cfg.final_img_filename = "vqa_score.png"
            return runner.run(
                run_dir,
                prompt_filename="vqa_score_prompt.txt",
                scores_filename="vqa_score_scores.json",
                summary_filename="vqa_score_result_summary.json",
            )
        return runner.run(run_dir / "vqa_score")

    if pipeline_name == "vanilla_sd":
        if flatten_output:
            return runner.run(
                run_dir,
                image_filename_template="vanilla_sd_{index:02d}.png",
                prompt_filename="vanilla_sd_prompt.txt",
                summary_filename="vanilla_sd_result_summary.json",
                **common_kwargs,
            )
        return runner.run(run_dir / "vanilla_sd", **common_kwargs)

    if pipeline_name == "flux1":
        if flatten_output:
            return runner.run(
                run_dir,
                image_filename_template="flux1_{index:02d}.png",
                prompt_filename="flux1_prompt.txt",
                summary_filename="flux1_result_summary.json",
                **common_kwargs,
            )
        return runner.run(run_dir / "flux1", **common_kwargs)

    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def save_prompt_summary(results: Dict[str, Dict[str, Any]], run_dir: Path) -> None:
    save_json(results, run_dir / "run_summary.json")


def release_runner_resources(runner) -> None:
    try:
        del runner
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass


def execute_selected_pipelines(
    cfg: DictConfig,
    run_dir: Path,
    prompt: str,
    flatten_output: bool = False,
) -> Dict[str, Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    for pipeline_name in get_selected_pipelines(cfg):
        runner = instantiate_pipeline_runner(cfg, pipeline_name)
        try:
            results[pipeline_name] = run_single_pipeline(
                runner=runner,
                cfg=cfg,
                pipeline_name=pipeline_name,
                run_dir=run_dir,
                prompt=prompt,
                flatten_output=flatten_output,
            )
            save_prompt_summary(results, run_dir)
        finally:
            release_runner_resources(runner)

    return results
