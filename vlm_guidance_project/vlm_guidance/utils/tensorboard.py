from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore[assignment]


log = logging.getLogger(__name__)

_SUMMARY_SCALAR_ALLOWLIST = {
    "final_score",
}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)


def _add_scalar(writer: SummaryWriter, tag: str, value: Any, global_step: Optional[int]) -> None:
    scalar_value = float(value) if isinstance(value, bool) else value
    if global_step is None:
        writer.add_scalar(tag, scalar_value)
    else:
        writer.add_scalar(tag, scalar_value, global_step=global_step)


def _log_scalar_tree(
    writer: SummaryWriter,
    value: Any,
    tag_prefix: str,
    global_step: Optional[int] = None,
) -> None:
    if value is None:
        return

    if _is_scalar(value):
        _add_scalar(writer, tag_prefix, value, global_step)
        return

    if isinstance(value, dict):
        next_step = global_step
        denoise_step = value.get("denoise_step")
        if _is_scalar(denoise_step):
            next_step = int(denoise_step)

        for key, nested_value in value.items():
            if isinstance(nested_value, str):
                continue
            nested_tag = f"{tag_prefix}/{key}" if tag_prefix else key
            _log_scalar_tree(writer, nested_value, nested_tag, next_step)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            return

        if all(_is_scalar(item) for item in value):
            if len(value) == 1:
                _add_scalar(writer, tag_prefix, value[0], global_step)
            else:
                for idx, item in enumerate(value):
                    _add_scalar(writer, f"{tag_prefix}/{idx}", item, global_step)
            return

        if all(isinstance(item, dict) for item in value):
            for idx, item in enumerate(value):
                nested_step = global_step
                nested_tag = tag_prefix

                if tag_prefix.endswith("/steps"):
                    step_value = item.get("denoise_step")
                    if _is_scalar(step_value):
                        nested_step = int(step_value)
                elif tag_prefix.endswith("/gd_stats"):
                    gd_iter = item.get("gd_iter", idx + 1)
                    if _is_scalar(gd_iter):
                        nested_tag = f"{tag_prefix}/gd_iter_{int(gd_iter):02d}"
                    else:
                        nested_tag = f"{tag_prefix}/item_{idx:02d}"
                else:
                    nested_tag = f"{tag_prefix}/item_{idx:02d}"

                _log_scalar_tree(writer, item, nested_tag, nested_step)
            return

        for idx, item in enumerate(value):
            _log_scalar_tree(writer, item, f"{tag_prefix}/{idx}", global_step)


def _log_flat_scalars(
    writer: SummaryWriter,
    values: dict[str, Any],
    tag_prefix: str,
    global_step: Optional[int] = None,
    skip_keys: Optional[set[str]] = None,
) -> None:
    skip_keys = skip_keys or set()
    for key, value in values.items():
        if key in skip_keys or isinstance(value, str) or value is None:
            continue
        if _is_scalar(value):
            _add_scalar(writer, f"{tag_prefix}/{key}", value, global_step)
        elif isinstance(value, (list, tuple)) and value and all(_is_scalar(item) for item in value):
            if len(value) == 1:
                _add_scalar(writer, f"{tag_prefix}/{key}", value[0], global_step)
            else:
                for idx, item in enumerate(value):
                    _add_scalar(writer, f"{tag_prefix}/{key}/{idx}", item, global_step)


def _log_step_records(writer: SummaryWriter, steps: list[dict[str, Any]], tag_prefix: str) -> None:
    for step_index, step in enumerate(steps):
        denoise_step = step.get("denoise_step")
        global_step = int(denoise_step) if _is_scalar(denoise_step) else step_index

        _log_flat_scalars(
            writer,
            step,
            tag_prefix=f"{tag_prefix}/steps",
            global_step=global_step,
            skip_keys={"denoise_step", "gd_stats", "debug"},
        )

        gd_stats = step.get("gd_stats")
        if isinstance(gd_stats, list):
            for gd_index, gd_item in enumerate(gd_stats):
                if not isinstance(gd_item, dict):
                    continue
                gd_iter = gd_item.get("gd_iter", gd_index + 1)
                gd_suffix = f"iter_{int(gd_iter):02d}" if _is_scalar(gd_iter) else f"item_{gd_index:02d}"

                _log_flat_scalars(
                    writer,
                    gd_item,
                    tag_prefix=f"{tag_prefix}/gd/{gd_suffix}",
                    global_step=global_step,
                    skip_keys={"denoise_step", "gd_iter", "debug"},
                )

                debug_block = gd_item.get("debug")
                if isinstance(debug_block, dict):
                    for tensor_name, tensor_stats in debug_block.items():
                        if not isinstance(tensor_stats, dict):
                            continue
                        _log_flat_scalars(
                            writer,
                            tensor_stats,
                            tag_prefix=f"{tag_prefix}/debug/{tensor_name}/{gd_suffix}",
                            global_step=global_step,
                            skip_keys={"name", "shape", "dtype"},
                        )


def _log_summary_dict(writer: SummaryWriter, summary: dict[str, Any], tag_prefix: str) -> None:
    for key, value in summary.items():
        if key in {"run", "guidance", "steps", "image_paths", "final_image_path"}:
            continue
        if isinstance(value, str) or value is None:
            continue
        if _is_scalar(value):
            if key not in _SUMMARY_SCALAR_ALLOWLIST:
                continue
            _add_scalar(writer, f"{tag_prefix}/summary/{key}", value, None)

    steps = summary.get("steps")
    if isinstance(steps, list) and all(isinstance(step, dict) for step in steps):
        _log_step_records(writer, steps, tag_prefix=tag_prefix)


def log_scalars_to_tensorboard(
    summary: Any,
    log_dir: str | Path,
    tag_prefix: str,
) -> None:
    if SummaryWriter is None:
        log.warning("TensorBoard is unavailable; install 'tensorboard' to enable event logging.")
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        if isinstance(summary, dict):
            _log_summary_dict(writer, summary, tag_prefix=tag_prefix)
        else:
            _log_scalar_tree(writer, summary, tag_prefix=tag_prefix)
        writer.flush()
    finally:
        writer.close()


class TensorBoardRunLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Optional[SummaryWriter]
        if SummaryWriter is None:
            log.warning("TensorBoard is unavailable; install 'tensorboard' to enable event logging.")
            self.writer = None
        else:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def add_image(self, tag: str, image: torch.Tensor, global_step: Optional[int] = None) -> None:
        if self.writer is None:
            return
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(image.shape)}")
        image = image.clamp(0, 1)
        if global_step is None:
            self.writer.add_image(tag, image)
        else:
            self.writer.add_image(tag, image, global_step=global_step)

    def add_text(self, tag: str, text: str, global_step: Optional[int] = None) -> None:
        if self.writer is None:
            return
        if global_step is None:
            self.writer.add_text(tag, text)
        else:
            self.writer.add_text(tag, text, global_step=global_step)

    def log_summary_scalars(self, summary: Any, tag_prefix: str) -> None:
        if self.writer is None:
            return
        if isinstance(summary, dict):
            _log_summary_dict(self.writer, summary, tag_prefix=tag_prefix)
        else:
            _log_scalar_tree(self.writer, summary, tag_prefix=tag_prefix)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
