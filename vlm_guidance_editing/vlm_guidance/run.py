from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vlm_guidance_editing.vlm_guidance.null_text_inversion import (
    IMAGE_EXTENSIONS,
    InversionArtifacts,
    NullTextInversionPipeline,
    load_image,
    save_artifacts,
)
from vlm_guidance_project.vlm_guidance.utils.images import save_diff_image, save_image_tensor

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR_NAMES = {
    "pipeline_null_text_inversion": "null_text_edit",
    "pipeline_vlm_guided_editing": "vlm_guided_null_text_edit",
}


@dataclass
class EditTask:
    run_index: int
    sample_id: str
    sample_dir: Path
    edit_prompt_key: str
    source_prompt: str
    edit_prompt: Optional[str]
    save_dir: Path


def get_selected_pipelines(cfg: DictConfig) -> List[str]:
    run_cfg = cfg.run
    use_null_text = bool(run_cfg.get("pipeline_null_text_inversion", run_cfg.get("null_text_inversion", False)))
    use_vlm_guided = bool(run_cfg.get("pipeline_vlm_guided_editing", run_cfg.get("vlm_guided_editing", False)))

    selected: List[str] = []
    if use_vlm_guided:
        selected.append("pipeline_vlm_guided_editing")
    if use_null_text:
        selected.append("pipeline_null_text_inversion")
    if not selected:
        raise ValueError(
            "At least one pipeline must be enabled: "
            "run.pipeline_null_text_inversion or run.pipeline_vlm_guided_editing."
        )
    return selected


def _resolve_from_project_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _find_image(sample_dir: Path) -> Path:
    image_path = next((p for p in sorted(sample_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS), None)
    if image_path is None:
        raise FileNotFoundError(f"No image file found in {sample_dir}")
    return image_path


def _read_required_prompt(sample_dir: Path, filename: str) -> str:
    prompt_path = sample_dir / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return text


def safe_prompt_dirname(index: int, prompt: str, max_len: int = 80) -> str:
    prompt_clean = prompt.strip().lower()
    prompt_clean = re.sub(r"\s+", "_", prompt_clean)
    prompt_clean = re.sub(r"[^a-zA-Z0-9а-яА-Я_=-]+", "", prompt_clean)
    prompt_clean = prompt_clean[:max_len].strip("_") or "prompt"
    return f"{index:04d}_{prompt_clean}"


def _resolve_edit_prompt_file(sample_dir: Path, mode: str) -> Optional[Path]:
    filename_candidates = {
        "object": ["prompt_object.txt", "edit_object_prompt.txt"],
        "background": ["prompt_background.txt", "edit_background_prompt.txt"],
        "color": ["prompt_color.txt"],
        "style": ["prompt_style.txt"],
        "texture": ["prompt_texture.txt"],
        "material": ["prompt_material.txt", "edit_material_prompt.txt"],
        "edit_prompt_txt": ["edit_prompt.txt"],
    }
    for candidate in filename_candidates[mode]:
        prompt_path = sample_dir / candidate
        if prompt_path.exists():
            return prompt_path
    return None


def _resolve_edit_prompts(sample_dir: Path, cfg: DictConfig) -> List[tuple[str, Optional[str]]]:
    mode = str(cfg.run.edit_prompt_mode).strip().lower()

    if mode == "none":
        return [("none", None)]
    if mode == "custom":
        value = cfg.run.custom_edit_prompt
        if value is None or not str(value).strip():
            raise ValueError("run.custom_edit_prompt must be provided when run.edit_prompt_mode=custom")
        return [("custom", str(value).strip())]

    if mode == "all":
        prompt_files = sorted(sample_dir.glob("prompt_*.txt"))
        prompts: List[tuple[str, Optional[str]]] = []
        for prompt_path in prompt_files:
            text = prompt_path.read_text(encoding="utf-8").strip()
            if not text:
                log.warning("Edit prompt file is empty for sample '%s': %s. It will be skipped.", sample_dir.name, prompt_path)
                continue
            prompts.append((prompt_path.stem.removeprefix("prompt_"), text))
        return prompts

    supported_modes = ["object", "background", "color", "style", "texture", "material", "edit_prompt_txt"]
    if mode not in supported_modes:
        raise ValueError(
            "Unsupported run.edit_prompt_mode='{}'. Choose one of: {}".format(
                mode,
                ", ".join(sorted([*supported_modes, "all", "none", "custom"])),
            )
        )

    prompt_path = _resolve_edit_prompt_file(sample_dir, mode)
    if prompt_path is None:
        log.warning("Edit prompt file not found for sample '%s' in mode '%s'. Editing will be skipped.", sample_dir.name, mode)
        return []
    if not prompt_path.exists():
        return []

    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        log.warning("Edit prompt file is empty for sample '%s': %s. Editing will be skipped.", sample_dir.name, prompt_path)
        return []
    return [(mode, text)]


def _discover_samples(dataset_root: Path, sample_limit: Optional[int]) -> list[Path]:
    sample_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
    if sample_limit is not None:
        sample_dirs = sample_dirs[: int(sample_limit)]
    return sample_dirs


def _load_sample_inputs(sample_dir: Path, cfg: DictConfig):
    image_path = _find_image(sample_dir)
    source_prompt = _read_required_prompt(sample_dir, str(cfg.run.source_prompt_filename))
    resized_input = load_image(image_path, size=(int(cfg.inversion.image_size), int(cfg.inversion.image_size)))
    return image_path, source_prompt, resized_input


def _build_edit_tasks(dataset_root: Path, cfg: DictConfig, output_root_dir: Path) -> List[EditTask]:
    tasks: List[EditTask] = []
    for sample_dir in _discover_samples(dataset_root, cfg.run.sample_limit):
        source_prompt = _read_required_prompt(sample_dir, str(cfg.run.source_prompt_filename))
        edit_prompts = _resolve_edit_prompts(sample_dir, cfg)
        for edit_prompt_key, edit_prompt in edit_prompts:
            if edit_prompt is None:
                continue
            tasks.append(
                EditTask(
                    run_index=len(tasks),
                    sample_id=sample_dir.name,
                    sample_dir=sample_dir,
                    edit_prompt_key=edit_prompt_key,
                    source_prompt=source_prompt,
                    edit_prompt=edit_prompt,
                    save_dir=output_root_dir / safe_prompt_dirname(len(tasks), edit_prompt),
                )
            )
    return tasks


def _write_pipeline_prompt(output_dir: Path, prompt: Optional[str]) -> None:
    if prompt is None:
        return
    (output_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")


def _finalize_pipeline_output(output_dir: Path, final_image_path: Path, prompt: Optional[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_img = output_dir / "img.png"
    if final_image_path.resolve() != target_img.resolve():
        shutil.copy2(final_image_path, target_img)
    _write_pipeline_prompt(output_dir, prompt)
    return target_img


def _make_guided_debug_dirs(output_dir: Path, enabled: bool) -> Dict[str, Optional[Path]]:
    if not enabled:
        return {"xt": None, "x0": None, "xdiff": None, "x0diff": None}
    dirs = {
        "xt": output_dir / "xt_gd",
        "x0": output_dir / "x0_gd",
        "xdiff": output_dir / "xdiff_gd",
        "x0diff": output_dir / "x0diff_gd",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _run_null_text_inversion_pipeline(
    *,
    cfg: DictConfig,
    pipeline: NullTextInversionPipeline,
    image_path: Path,
    resized_input: "Any",
    source_prompt: str,
    edit_prompt: Optional[str],
    output_dir: Path,
) -> Dict[str, Any]:
    artifacts = pipeline.run(
        image_bchw=resized_input,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        num_null_text_steps=int(cfg.inversion.num_null_text_steps),
        null_text_lr=float(cfg.inversion.null_text_lr),
        early_stop_eps=float(cfg.inversion.early_stop_eps),
    )
    save_artifacts(output_dir, image_path, artifacts, resized_input)
    edited_candidates = sorted(output_dir.glob("edited_*.png"))
    final_image_path = edited_candidates[0] if edited_candidates else output_dir / "reconstruction.png"
    final_image_path = _finalize_pipeline_output(output_dir, final_image_path, edit_prompt)
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "final_image": str(final_image_path),
    }


def _run_vlm_guided_editing_pipeline(
    *,
    cfg: DictConfig,
    pipeline: NullTextInversionPipeline,
    scorer: Any,
    image_path: Path,
    resized_input: "Any",
    source_prompt: str,
    edit_prompt: Optional[str],
    output_dir: Path,
) -> Dict[str, Any]:
    if edit_prompt is None:
        return {
            "status": "skipped",
            "reason": "No edit prompt for selected mode; guided editing requires edit prompt.",
            "output_dir": str(output_dir),
        }

    image_latent = pipeline.encode_image(resized_input.float().cpu())
    cond_embeddings = pipeline.encode_text(source_prompt)
    base_uncond_embeddings = pipeline.encode_text("")
    ddim_latents = pipeline.ddim_inversion(image_latent, cond_embeddings)
    null_text_embeddings, loss_history = pipeline.null_text_optimization(
        ddim_latents=ddim_latents,
        cond_embeddings=cond_embeddings,
        base_uncond_embeddings=base_uncond_embeddings,
        num_inner_steps=int(cfg.inversion.num_null_text_steps),
        lr=float(cfg.inversion.null_text_lr),
        early_stop_eps=float(cfg.inversion.early_stop_eps),
    )

    start_latent = ddim_latents[-1]
    reconstruction_latent = pipeline.sample_with_uncond_sequence(start_latent, cond_embeddings, null_text_embeddings)
    reconstruction = pipeline.decode_latents(reconstruction_latent)

    edit_embeddings = pipeline.encode_text(edit_prompt)
    debug_enabled = bool(cfg.algorithm.save_debug_tensors)
    debug_dirs = _make_guided_debug_dirs(output_dir, debug_enabled)
    n_digits = max(4, len(str(len(pipeline.scheduler.timesteps))))

    def _guided_debug_hook(payload: Dict[str, Any]) -> None:
        if not debug_enabled:
            return
        step = int(payload["denoise_step"])
        gd_iter = payload.get("gd_iter")
        base = f"t_{step:0{n_digits}d}"

        event = payload["event"]
        if event == "gd_iter_before_update":
            if gd_iter is None:
                return
            save_image_tensor(payload["x_t_img_before"], debug_dirs["xt"] / f"{base}_gd_{gd_iter:02d}_before_gd_step.png")
            save_image_tensor(payload["x0_img_before"], debug_dirs["x0"] / f"{base}_gd_{gd_iter:02d}_before_gd_step.png")
            return

        if event == "gd_iter_after_update":
            if gd_iter is None:
                return
            save_diff_image(
                payload["x_t_img_before"],
                payload["x_t_img_after"],
                debug_dirs["xdiff"] / f"{base}_gd_{gd_iter:02d}.png",
            )
            return

        if event == "step_done_gd":
            save_image_tensor(payload["x_t_img_after_gd"], debug_dirs["xt"] / f"{base}_done_gd_step.png")
            return

        if event == "step_before_denoise":
            x0_before = payload.get("x0_img_before_denoise")
            if x0_before is not None:
                save_image_tensor(x0_before, debug_dirs["x0"] / f"{base}_before_denoise.png")
            x0_before_guidance = payload.get("x0_img_before_guidance")
            if x0_before is not None and x0_before_guidance is not None:
                save_diff_image(
                    x0_before_guidance,
                    x0_before,
                    debug_dirs["x0diff"] / f"{base}_cumulative_before_denoise.png",
                )
            return

    guided_latent, guided_null_text_embeddings, guided_step_records = pipeline.sample_with_vlm_guidance(
        start_latent=start_latent,
        cond_embeddings=edit_embeddings,
        uncond_embeddings_sequence=null_text_embeddings,
        scorer=scorer,
        prompt_for_loss=edit_prompt,
        gd_steps=int(cfg.guided.gd_steps),
        gd_only_first_k_steps=None if cfg.algorithm.gd_only_first_k_steps is None else int(cfg.algorithm.gd_only_first_k_steps),
        zt_optimizing=bool(cfg.guided.zt_optimizing),
        zt_lr=float(cfg.guided.zt_lr),
        null_text_emb_optimizing=bool(cfg.guided.null_text_emb_optimizing),
        null_text_emb_lr=float(cfg.guided.null_text_emb_lr),
        normalize_grad=bool(cfg.guided.normalize_grad),
        clamp_grad_value=None if cfg.guided.clamp_grad_value is None else float(cfg.guided.clamp_grad_value),
        debug_hook=_guided_debug_hook if debug_enabled else None,
    )
    guided_image = pipeline.decode_latents(guided_latent)

    artifacts = InversionArtifacts(
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        image_latent=image_latent.float().cpu(),
        ddim_latents=list(ddim_latents),
        null_text_embeddings=list(guided_null_text_embeddings),
        null_text_loss_history=list(loss_history),
        reconstruction=reconstruction,
        edited=guided_image,
    )
    save_artifacts(output_dir, image_path, artifacts, resized_input)
    guided_image_path = output_dir / str(cfg.guided.final_img_filename)
    save_image_tensor(guided_image, guided_image_path)
    final_image_path = _finalize_pipeline_output(output_dir, guided_image_path, edit_prompt)

    (output_dir / str(cfg.guided.summary_filename)).write_text(
        json.dumps(
            {
                "pipeline_name": "vlm_guided_editing",
                "source_prompt": source_prompt,
                "edit_prompt": edit_prompt,
                "guided": OmegaConf.to_container(cfg.guided, resolve=True),
                "algorithm": OmegaConf.to_container(cfg.algorithm, resolve=True),
                "steps": guided_step_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "final_image": str(final_image_path),
        "guided_summary": str(output_dir / str(cfg.guided.summary_filename)),
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=False))

    dataset_root = _resolve_from_project_root(cfg.run.dataset_root)
    output_root_dir = _resolve_from_project_root(cfg.run.output_root_dir)
    summary_path = output_root_dir / str(cfg.run.summary_filename)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_root_dir.mkdir(parents=True, exist_ok=True)
    edit_tasks = _build_edit_tasks(dataset_root, cfg, output_root_dir)
    if not edit_tasks:
        raise ValueError(f"No edit tasks found in {dataset_root}")

    selected_pipelines = get_selected_pipelines(cfg)
    pipeline = NullTextInversionPipeline(
        model_id=cfg.model.model_id,
        device=cfg.model.device,
        weights_dtype=cfg.model.weights_dtype,
        num_ddim_steps=cfg.inversion.num_ddim_steps,
        guidance_scale=cfg.inversion.guidance_scale,
    )
    scorer = instantiate(cfg.scorer) if "pipeline_vlm_guided_editing" in selected_pipelines else None

    run_results: List[Dict[str, Any]] = []
    for task in edit_tasks:
        run_results.append({
            "prompt_index": task.run_index,
            "sample_id": task.sample_id,
            "edit_prompt_key": task.edit_prompt_key,
            "prompt": task.edit_prompt,
            "source_prompt": task.source_prompt,
            "status": "pending",
            "save_dir": str(task.save_dir),
            "pipelines": {},
        })

    stop_all = False
    for pipeline_name in selected_pipelines:
        if stop_all:
            break
        for task in tqdm(edit_tasks, desc=pipeline_name, dynamic_ncols=True):
            result_item = run_results[task.run_index]
            pipeline_output_dir = task.save_dir / PIPELINE_DIR_NAMES[pipeline_name]

            if cfg.run.skip_existing and (pipeline_output_dir / "img.png").exists():
                result_item["pipelines"][PIPELINE_DIR_NAMES[pipeline_name]] = {
                    "status": "skipped_existing",
                    "output_dir": str(pipeline_output_dir),
                }
                (task.save_dir / str(cfg.run.summary_filename)).write_text(
                    json.dumps(result_item["pipelines"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                continue

            try:
                image_path, source_prompt, resized_input = _load_sample_inputs(task.sample_dir, cfg)
                result_item["image_path"] = str(image_path)

                if pipeline_name == "pipeline_null_text_inversion":
                    result = _run_null_text_inversion_pipeline(
                        cfg=cfg,
                        pipeline=pipeline,
                        image_path=image_path,
                        resized_input=resized_input,
                        source_prompt=source_prompt,
                        edit_prompt=task.edit_prompt,
                        output_dir=pipeline_output_dir,
                    )
                elif pipeline_name == "pipeline_vlm_guided_editing":
                    if scorer is None:
                        raise RuntimeError("Scorer is not initialized for pipeline_vlm_guided_editing.")
                    result = _run_vlm_guided_editing_pipeline(
                        cfg=cfg,
                        pipeline=pipeline,
                        scorer=scorer,
                        image_path=image_path,
                        resized_input=resized_input,
                        source_prompt=source_prompt,
                        edit_prompt=task.edit_prompt,
                        output_dir=pipeline_output_dir,
                    )
                else:
                    raise ValueError(f"Unknown pipeline: {pipeline_name}")

                result["pipeline_name"] = PIPELINE_DIR_NAMES[pipeline_name]
                result_item["pipelines"][PIPELINE_DIR_NAMES[pipeline_name]] = result
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                log.exception(
                    "Failed sample '%s' prompt '%s' in pipeline '%s': %s",
                    task.sample_id,
                    task.edit_prompt_key,
                    pipeline_name,
                    error_msg,
                )
                result_item["pipelines"][PIPELINE_DIR_NAMES[pipeline_name]] = {
                    "status": "error",
                    "error": error_msg,
                    "output_dir": str(pipeline_output_dir),
                }
                if cfg.run.fail_fast:
                    stop_all = True
                    break
            (task.save_dir / str(cfg.run.summary_filename)).write_text(
                json.dumps(result_item["pipelines"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    for item in run_results:
        pipeline_statuses = [p.get("status", "unknown") for p in item.get("pipelines", {}).values()]
        if any(status == "error" for status in pipeline_statuses):
            item["status"] = "error"
        elif pipeline_statuses and all(status == "skipped_existing" for status in pipeline_statuses):
            item["status"] = "skipped_existing"
        elif pipeline_statuses:
            item["status"] = "ok"
        else:
            item["status"] = "pending"

    summary = {
        "dataset_root": str(dataset_root),
        "output_root_dir": str(output_root_dir),
        "selected_pipelines": [PIPELINE_DIR_NAMES[name] for name in selected_pipelines],
        "num_samples": len({task.sample_id for task in edit_tasks}),
        "num_edit_tasks": len(edit_tasks),
        "num_ok": sum(1 for x in run_results if x["status"] == "ok"),
        "num_error": sum(1 for x in run_results if x["status"] == "error"),
        "num_skipped_existing": sum(1 for x in run_results if x["status"] == "skipped_existing"),
        "results": run_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Finished. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
