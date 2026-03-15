from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vlm_guidance.execution import (
    get_selected_pipelines,
    instantiate_pipeline_runner,
    release_runner_resources,
    run_single_pipeline,
    save_prompt_summary,
)
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_from_project_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_prompts(path: str, skip_empty_lines: bool = True) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f]
    if skip_empty_lines:
        prompts = [p.strip() for p in raw if p.strip()]
    else:
        prompts = [p.strip() for p in raw]
    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")
    return prompts


def safe_prompt_dirname(index: int, prompt: str, max_len: int = 80) -> str:
    import re
    prompt_clean = prompt.strip().lower()
    prompt_clean = re.sub(r"\s+", "_", prompt_clean)
    prompt_clean = re.sub(r"[^a-zA-Z0-9а-яА-Я_=-]+", "", prompt_clean)
    prompt_clean = prompt_clean[:max_len].strip("_") or "prompt"
    return f"{index:04d}_{prompt_clean}"


@hydra.main(version_base=None, config_path="configs", config_name="batch_config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=False))
    prompts_file = _resolve_from_project_root(cfg.batch.prompts_file)
    output_root_dir = _resolve_from_project_root(cfg.batch.output_root_dir)

    prompts = read_prompts(str(prompts_file), skip_empty_lines=cfg.batch.skip_empty_lines)

    root_dir = output_root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    prompt_dirs = [root_dir / safe_prompt_dirname(i, prompt) for i, prompt in enumerate(prompts)]
    results: List[Dict] = []
    for i, (prompt, save_dir) in enumerate(zip(prompts, prompt_dirs)):
        save_dir.mkdir(parents=True, exist_ok=True)
        results.append({
            "prompt_index": i,
            "prompt": prompt,
            "save_dir": str(save_dir),
            "pipelines": {},
        })

    selected_pipelines = get_selected_pipelines(cfg)
    for pipeline_name in selected_pipelines:
        log.info("Running batch with pipeline '%s' over %d prompts", pipeline_name, len(prompts))
        runner = instantiate_pipeline_runner(cfg, pipeline_name)
        try:
            for i, prompt in enumerate(tqdm(prompts, desc=f"{pipeline_name}", total=len(prompts), dynamic_ncols=True)):
                save_dir = prompt_dirs[i]
                pipeline_result = run_single_pipeline(
                    runner=runner,
                    cfg=cfg,
                    pipeline_name=pipeline_name,
                    run_dir=save_dir,
                    prompt=prompt,
                )
                results[i]["pipelines"][pipeline_name] = pipeline_result
                save_prompt_summary(results[i]["pipelines"], save_dir)
        finally:
            release_runner_resources(runner)

    save_json(results, root_dir / "run_summary.json")
    log.info(
        "Completed %d prompts across %d pipelines. Summary saved to %s",
        len(prompts),
        len(selected_pipelines),
        root_dir / "run_summary.json",
    )


if __name__ == "__main__":
    main()
