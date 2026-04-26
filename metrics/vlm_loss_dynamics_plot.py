import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_OUTPUT = Path("metrics/vlm_loss_dynamics.png")
PIPELINE_COLORS = {
    "vanilla_sd": "#1f77b4",
    "vqa_score": "#d95f02",
}
PIPELINE_LABELS = {
    "vanilla_sd": "Vanilla SD",
    "vqa_score": "Vlm guided",
}
PIPELINES = tuple(PIPELINE_COLORS)


@dataclass(frozen=True)
class LossTrajectory:
    pipeline: str
    prompt_name: str
    x_values: list[float]
    y_values: list[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VLM loss dynamics for vanilla_sd and vqa_score result_summary.json "
            "files stored under prompt generation folders."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Directory with prompt subfolders, each containing vanilla_sd/ and vqa_score/ result summaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output plot image. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=14.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=8.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--show-means",
        action="store_true",
        help="Overlay bold mean trajectories for each pipeline.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _as_float(value: Any, context: str) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty list where numeric value was expected: {context}")
        value = value[0]
    if value is None:
        raise ValueError(f"Missing numeric value: {context}")
    return float(value)


def _step_x_value(step: dict[str, Any], fallback_index: int) -> float:
    return _as_float(step.get("denoise_step", fallback_index), "denoise_step")


def _extract_vanilla_sd_trajectory(summary: dict[str, Any]) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    for index, step in enumerate(summary.get("steps", [])):
        if "vlm_loss" not in step:
            continue
        x_values.append(_step_x_value(step, index))
        y_values.append(_as_float(step["vlm_loss"], f"steps[{index}].vlm_loss"))

    if not y_values:
        raise ValueError("No vanilla_sd vlm_loss values found")
    return x_values, y_values


def _extract_vqa_score_trajectory(summary: dict[str, Any]) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    for step_index, step in enumerate(summary.get("steps", [])):
        gd_stats = step.get("gd_stats", [])
        for gd_stat in gd_stats:
            if int(gd_stat.get("gd_iter", -1)) != 1:
                continue
            if "loss_before_update" not in gd_stat:
                continue
            x_values.append(_step_x_value(step, step_index))
            y_values.append(
                _as_float(
                    gd_stat["loss_before_update"],
                    f"steps[{step_index}].gd_stats[gd_iter=1].loss_before_update",
                )
            )
            break

    if not y_values:
        raise ValueError("No vqa_score gd_iter=1 loss_before_update values found")
    return x_values, y_values


def _filter_trajectory_to_x_values(
    x_values: list[float],
    y_values: list[float],
    allowed_x_values: list[float],
) -> tuple[list[float], list[float]]:
    allowed = set(allowed_x_values)
    filtered_x_values: list[float] = []
    filtered_y_values: list[float] = []

    for x_value, y_value in zip(x_values, y_values):
        if x_value not in allowed:
            continue
        filtered_x_values.append(x_value)
        filtered_y_values.append(y_value)

    if not filtered_y_values:
        raise ValueError("No vanilla_sd vlm_loss values match vqa_score denoise steps")
    return filtered_x_values, filtered_y_values


def _prompt_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    prompt_dirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and any((path / pipeline).is_dir() for pipeline in PIPELINES)
    )
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt subdirectories found under {input_root}")
    return prompt_dirs


def _collect_trajectories(input_root: Path) -> list[LossTrajectory]:
    trajectories: list[LossTrajectory] = []
    missing_paths: list[Path] = []
    failed_paths: list[tuple[Path, str]] = []

    for prompt_dir in _prompt_dirs(input_root):
        vanilla_path = prompt_dir / "vanilla_sd" / "result_summary.json"
        vqa_path = prompt_dir / "vqa_score" / "result_summary.json"

        if not vanilla_path.is_file():
            missing_paths.append(vanilla_path)
            continue
        if not vqa_path.is_file():
            missing_paths.append(vqa_path)
            continue

        try:
            vqa_x_values, vqa_y_values = _extract_vqa_score_trajectory(_load_json(vqa_path))
        except (KeyError, TypeError, ValueError) as exc:
            failed_paths.append((vqa_path, str(exc)))
            continue

        try:
            vanilla_x_values, vanilla_y_values = _extract_vanilla_sd_trajectory(_load_json(vanilla_path))
            vanilla_x_values, vanilla_y_values = _filter_trajectory_to_x_values(
                vanilla_x_values,
                vanilla_y_values,
                allowed_x_values=vqa_x_values,
            )
        except (KeyError, TypeError, ValueError) as exc:
            failed_paths.append((vanilla_path, str(exc)))
            continue

        trajectories.extend(
            [
                LossTrajectory(
                    pipeline="vanilla_sd",
                    prompt_name=prompt_dir.name,
                    x_values=vanilla_x_values,
                    y_values=vanilla_y_values,
                ),
                LossTrajectory(
                    pipeline="vqa_score",
                    prompt_name=prompt_dir.name,
                    x_values=vqa_x_values,
                    y_values=vqa_y_values,
                ),
            ]
        )

    if missing_paths:
        print(f"Warning: skipped {len(missing_paths)} missing result_summary.json files")
    if failed_paths:
        for path, reason in failed_paths:
            print(f"Warning: skipped {path}: {reason}")

    if not trajectories:
        raise ValueError(f"No VLM loss trajectories collected from {input_root}")
    return trajectories


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", alpha=0.18, color="#36454f", linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.08, color="#36454f", linewidth=0.5)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, color="#9aa4ad")
    ax.tick_params(axis="both", which="minor", length=2, color="#c4ccd3")


def _mean_trajectory(trajectories: list[LossTrajectory]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    for trajectory in trajectories:
        for x_value, y_value in zip(trajectory.x_values, trajectory.y_values):
            grouped.setdefault(float(x_value), []).append(float(y_value))

    x_values = np.asarray(sorted(grouped), dtype=float)
    y_values = np.asarray([np.mean(grouped[x_value]) for x_value in x_values], dtype=float)
    return x_values, y_values


def _plot_trajectories(
    trajectories: list[LossTrajectory],
    output_path: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
    show_means: bool,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.labelcolor": "#1f2933",
            "axes.titleweight": "bold",
            "xtick.color": "#1f2933",
            "ytick.color": "#1f2933",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="#ffffff")
    grouped_by_pipeline: dict[str, list[LossTrajectory]] = {}

    for trajectory in trajectories:
        grouped_by_pipeline.setdefault(trajectory.pipeline, []).append(trajectory)
        color = PIPELINE_COLORS.get(trajectory.pipeline, "#606c38")
        ax.plot(
            trajectory.x_values,
            trajectory.y_values,
            color=color,
            linewidth=1.45,
            alpha=0.24,
            marker="o",
            markersize=2.5,
            markeredgewidth=0.0,
        )

    if show_means:
        for pipeline, pipeline_trajectories in sorted(grouped_by_pipeline.items()):
            mean_x, mean_y = _mean_trajectory(pipeline_trajectories)
            ax.plot(
                mean_x,
                mean_y,
                color=PIPELINE_COLORS.get(pipeline, "#606c38"),
                linewidth=3.4,
                alpha=0.95,
                label=f"{PIPELINE_LABELS.get(pipeline, pipeline)} mean",
                solid_capstyle="round",
                zorder=5,
            )

    pipeline_handles = [
        Line2D(
            [0],
            [0],
            color=PIPELINE_COLORS.get(pipeline, "#606c38"),
            linewidth=2.8,
            label=f"{PIPELINE_LABELS.get(pipeline, pipeline)} ({len(pipeline_trajectories)} runs)",
        )
        for pipeline, pipeline_trajectories in sorted(grouped_by_pipeline.items())
    ]
    if show_means:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    else:
        ax.legend(
            handles=pipeline_handles,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
        )

    ax.set_title("VLM Loss Dynamics", fontsize=20, fontweight="bold", color="#111827", pad=14)
    ax.set_xlabel("Denoise step")
    ax.set_ylabel("VLM loss")
    ax.margins(x=0.02, y=0.08)
    ax.set_xticks(sorted({x_value for trajectory in trajectories for x_value in trajectory.x_values}))
    _style_axes(ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.figure_width <= 0 or args.figure_height <= 0:
        raise ValueError("Figure size must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    trajectories = _collect_trajectories(args.input_root)
    _plot_trajectories(
        trajectories=trajectories,
        output_path=args.output,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        show_means=args.show_means,
    )

    counts = {
        pipeline: sum(trajectory.pipeline == pipeline for trajectory in trajectories)
        for pipeline in sorted({trajectory.pipeline for trajectory in trajectories})
    }
    print(f"Collected {len(trajectories)} trajectories from {args.input_root}")
    for pipeline, count in counts.items():
        print(f"- {pipeline}: {count}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
