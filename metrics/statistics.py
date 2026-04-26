import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_ROOT = Path("vlm_guidance_project/subset_generations")
DEFAULT_OUTPUT = Path("metrics/grad_norm_raw_trajectories.png")
DEFAULT_SPLIT_PERCENTILE = 0.2


Series = tuple[list[float], list[float]]
SeriesExtractor = Callable[[dict], Series]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot guidance and diffusion statistics for all subset_generations prompt folders."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing prompt subfolders with vqa_score/result_summary.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output PNG file for grad_norm_raw trajectories.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=21.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=8.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--split-percentile",
        type=float,
        default=DEFAULT_SPLIT_PERCENTILE,
        help="Fraction of runs to include in the best/worst final VLM loss split plot, for example 0.2 for 20%%.",
    )
    return parser.parse_args()


def _style_axes(ax) -> None:
    ax.set_facecolor("#fffaf2")
    ax.grid(True, alpha=0.24, color="#7f8c69", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5f6f52")
    ax.spines["bottom"].set_color("#5f6f52")


def _result_paths(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    paths = sorted(input_root.glob("*/vqa_score/result_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No result_summary.json files found under {input_root}")
    return paths


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_grad_norm_series(summary: dict) -> Series:
    x_values: list[float] = []
    y_values: list[float] = []
    guidance_step_index = 0

    for step in summary.get("steps", []):
        for gd_stat in step.get("gd_stats", []):
            grad_norm_raw = gd_stat.get("grad_norm_raw")
            if grad_norm_raw is None:
                continue
            x_values.append(float(guidance_step_index))
            y_values.append(float(grad_norm_raw))
            guidance_step_index += 1

    if not x_values:
        raise ValueError("No grad_norm_raw values found in result summary")

    return x_values, y_values


def _extract_loss_series(summary: dict) -> Series:
    x_values: list[float] = []
    y_values: list[float] = []
    guidance_step_index = 0

    for step in summary.get("steps", []):
        for gd_stat in step.get("gd_stats", []):
            loss_before_update = gd_stat.get("loss_before_update")
            if loss_before_update is None:
                continue
            x_values.append(float(guidance_step_index))
            y_values.append(float(loss_before_update))
            guidance_step_index += 1

    if not x_values:
        raise ValueError("No loss_before_update values found in result summary")

    return x_values, y_values


def _extract_final_loss(summary: dict) -> float:
    final_loss: float | None = None

    for step in summary.get("steps", []):
        for gd_stat in step.get("gd_stats", []):
            loss_before_update = gd_stat.get("loss_before_update")
            if loss_before_update is None:
                continue
            final_loss = float(loss_before_update)

    if final_loss is None:
        raise ValueError("No loss_before_update values found in result summary")

    return final_loss


def _extract_denoise_step_norm_series(summary: dict) -> Series:
    x_values: list[float] = []
    y_values: list[float] = []
    diffusion_step_index = 0

    for step in summary.get("steps", []):
        denoise_step_norm = step.get("denoise_step_norm")
        if denoise_step_norm is None:
            continue

        denoise_step = step.get("denoise_step")
        if denoise_step is None:
            denoise_step = diffusion_step_index

        x_values.append(float(denoise_step))
        y_values.append(float(denoise_step_norm))
        diffusion_step_index += 1

    if not x_values:
        raise ValueError("No denoise_step_norm values found in result summary")

    return x_values, y_values


def _extract_guidance_block_starts(summary: dict) -> list[float]:
    starts: list[float] = []
    guidance_step_index = 0

    for step in summary.get("steps", []):
        step_started = False
        for gd_stat in step.get("gd_stats", []):
            has_guidance_value = (
                gd_stat.get("grad_norm_raw") is not None
                or gd_stat.get("loss_before_update") is not None
            )
            if not has_guidance_value:
                continue
            if not step_started:
                starts.append(float(guidance_step_index))
                step_started = True
            guidance_step_index += 1

    return starts


def _extract_guidance_block_tick_map(summary: dict) -> dict[float, str]:
    tick_map: dict[float, str] = {}
    guidance_step_index = 0

    for step_index, step in enumerate(summary.get("steps", [])):
        step_started = False
        denoise_step = step.get("denoise_step")
        if denoise_step is None:
            denoise_step = step_index

        for gd_stat in step.get("gd_stats", []):
            has_guidance_value = (
                gd_stat.get("grad_norm_raw") is not None
                or gd_stat.get("loss_before_update") is not None
            )
            if not has_guidance_value:
                continue
            if not step_started:
                tick_map[float(guidance_step_index)] = str(denoise_step)
                step_started = True
            guidance_step_index += 1

    return tick_map


def _label_from_result_path(path: Path, input_root: Path) -> str:
    prompt_dir = path.relative_to(input_root).parts[0]
    return prompt_dir


def _collect_series(input_root: Path, extractor: SeriesExtractor) -> dict[str, Series]:
    series: dict[str, Series] = {}
    for result_path in _result_paths(input_root):
        summary = _load_json(result_path)
        label = _label_from_result_path(result_path, input_root)
        series[label] = extractor(summary)
    return series


def _collect_guidance_block_starts(input_root: Path) -> np.ndarray:
    starts: set[float] = set()
    for result_path in _result_paths(input_root):
        summary = _load_json(result_path)
        starts.update(_extract_guidance_block_starts(summary))
    return np.asarray(sorted(starts), dtype=float)


def _collect_guidance_block_ticks(input_root: Path) -> tuple[np.ndarray, list[str]]:
    tick_map: dict[float, str] = {}
    for result_path in _result_paths(input_root):
        summary = _load_json(result_path)
        for x_value, label in _extract_guidance_block_tick_map(summary).items():
            tick_map.setdefault(x_value, label)

    sorted_positions = np.asarray(sorted(tick_map.keys()), dtype=float)
    labels = [tick_map[x_value] for x_value in sorted_positions]
    return sorted_positions, labels


def _mean_series_by_x(series: dict[str, Series]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    for x_values, y_values in series.values():
        for x_value, y_value in zip(x_values, y_values):
            grouped.setdefault(float(x_value), []).append(float(y_value))

    sorted_x = np.asarray(sorted(grouped.keys()), dtype=float)
    mean_y = np.asarray([np.mean(grouped[x_value]) for x_value in sorted_x], dtype=float)
    return sorted_x, mean_y


def _plot_series(
    series: dict[str, Series],
    output_path: Path,
    figure_width: float,
    figure_height: float,
    title: str,
    xlabel: str,
    ylabel: str,
    mean_label: str,
    log_scale: bool = False,
    vertical_lines: np.ndarray | None = None,
    x_tick_positions: np.ndarray | None = None,
    x_tick_labels: list[str] | None = None,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f3eb",
            "axes.facecolor": "#fffaf2",
            "axes.labelcolor": "#283618",
            "xtick.color": "#283618",
            "ytick.color": "#283618",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="#f7f3eb")
    cmap = plt.get_cmap("tab10", len(series))

    if vertical_lines is not None:
        for x_value in vertical_lines:
            ax.axvline(x_value, color="#606c38", linewidth=0.8, alpha=0.12, zorder=0)

    for index, (label, (x_values, y_values)) in enumerate(series.items()):
        color = cmap(index)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            alpha=0.3,
            color=color,
            label=label,
        )

    mean_x, mean_y = _mean_series_by_x(series)
    ax.plot(
        mean_x,
        mean_y,
        linewidth=3.0,
        alpha=0.95,
        color="#bc6c25",
        label=mean_label,
        zorder=5,
    )

    if x_tick_positions is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=0)

    ax.set_title(title, fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style_axes(ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _split_grad_norm_series_by_final_loss(
    input_root: Path,
    split_percentile: float = DEFAULT_SPLIT_PERCENTILE,
) -> tuple[dict[str, Series], dict[str, Series]]:
    if not 0 < split_percentile < 0.5:
        raise ValueError("split_percentile must be between 0 and 0.5")

    collected: list[tuple[str, Series, float]] = []
    for result_path in _result_paths(input_root):
        summary = _load_json(result_path)
        label = _label_from_result_path(result_path, input_root)
        grad_series = _extract_grad_norm_series(summary)
        final_loss = _extract_final_loss(summary)
        collected.append((label, grad_series, final_loss))

    if not collected:
        raise ValueError("No grad_norm_raw trajectories available for split plot")

    split_count = max(1, int(np.ceil(len(collected) * split_percentile)))
    collected.sort(key=lambda item: item[2])

    lower_group = {
        label: series
        for label, series, _ in collected[:split_count]
    }
    upper_group = {
        label: series
        for label, series, _ in collected[-split_count:]
    }
    return lower_group, upper_group


def _split_output_path_for_metric(base_output: Path) -> Path:
    return base_output.with_name(f"grad_norm_raw_split_by_final_loss{base_output.suffix}")


def _plot_split_series(
    lower_series: dict[str, Series],
    upper_series: dict[str, Series],
    output_path: Path,
    figure_width: float,
    figure_height: float,
    title: str,
    xlabel: str,
    ylabel: str,
    lower_label: str,
    upper_label: str,
    log_scale: bool = False,
    vertical_lines: np.ndarray | None = None,
    x_tick_positions: np.ndarray | None = None,
    x_tick_labels: list[str] | None = None,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f3eb",
            "axes.facecolor": "#fffaf2",
            "axes.labelcolor": "#283618",
            "xtick.color": "#283618",
            "ytick.color": "#283618",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="#f7f3eb")

    if vertical_lines is not None:
        for x_value in vertical_lines:
            ax.axvline(x_value, color="#606c38", linewidth=0.8, alpha=0.12, zorder=0)

    group_configs = [
        (lower_series, "#2a9d8f", lower_label),
        (upper_series, "#d62828", upper_label),
    ]

    for series, color, mean_label in group_configs:
        for x_values, y_values in series.values():
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                alpha=0.18,
                color=color,
            )

        mean_x, mean_y = _mean_series_by_x(series)
        ax.plot(
            mean_x,
            mean_y,
            linewidth=3.0,
            alpha=0.95,
            color=color,
            label=mean_label,
            zorder=5,
        )

    if x_tick_positions is not None and x_tick_labels is not None:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=0)

    ax.set_title(title, fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style_axes(ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _output_path_for_metric(base_output: Path, metric_name: str) -> Path:
    if metric_name == "grad_norm_raw":
        return base_output
    return base_output.with_name(f"{metric_name}_trajectories{base_output.suffix}")


def main() -> None:
    args = _parse_args()
    if args.figure_width <= 0 or args.figure_height <= 0:
        raise ValueError("Figure size must be positive")
    if not 0 < args.split_percentile < 0.5:
        raise ValueError("--split-percentile must be between 0 and 0.5")

    result_paths = _result_paths(args.input_root)
    guidance_block_starts = _collect_guidance_block_starts(args.input_root)
    guidance_tick_positions, guidance_tick_labels = _collect_guidance_block_ticks(args.input_root)
    metric_configs = [
        {
            "metric_name": "grad_norm_raw",
            "extractor": _extract_grad_norm_series,
            "title": "grad_norm_raw Trajectories",
            "xlabel": "denoise_step",
            "ylabel": "grad_norm_raw",
            "mean_label": "mean grad_norm_raw across runs",
            "log_scale": True,
            "vertical_lines": guidance_block_starts,
            "x_tick_positions": guidance_tick_positions,
            "x_tick_labels": guidance_tick_labels,
        },
        {
            "metric_name": "denoise_step_norm",
            "extractor": _extract_denoise_step_norm_series,
            "title": "denoise_step_norm Trajectories",
            "xlabel": "diffusion_step",
            "ylabel": "denoise_step_norm",
            "mean_label": "mean denoise_step_norm across runs",
            "log_scale": False,
            "vertical_lines": None,
            "x_tick_positions": None,
            "x_tick_labels": None,
        },
        {
            "metric_name": "loss_before_update",
            "extractor": _extract_loss_series,
            "title": "loss_before_update Trajectories",
            "xlabel": "denoise_step",
            "ylabel": "loss_before_update",
            "mean_label": "mean loss_before_update across runs",
            "log_scale": False,
            "vertical_lines": guidance_block_starts,
            "x_tick_positions": guidance_tick_positions,
            "x_tick_labels": guidance_tick_labels,
        },
    ]

    print(f"Found {len(result_paths)} result_summary.json files")
    print(f"Computed {len(guidance_block_starts)} guidance block boundaries from JSON data")
    for config in metric_configs:
        series = _collect_series(args.input_root, config["extractor"])
        output_path = _output_path_for_metric(args.output, config["metric_name"])
        _plot_series(
            series=series,
            output_path=output_path,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            title=config["title"],
            xlabel=config["xlabel"],
            ylabel=config["ylabel"],
            mean_label=config["mean_label"],
            log_scale=config["log_scale"],
            vertical_lines=config["vertical_lines"],
            x_tick_positions=config["x_tick_positions"],
            x_tick_labels=config["x_tick_labels"],
        )
        print(f"Plotted {config['metric_name']} for {len(series)} runs")
        for label, (_, y_values) in series.items():
            print(f"- {label}: {len(y_values)} values")
        print(f"Saved plot to {output_path}")

    lower_split_series, upper_split_series = _split_grad_norm_series_by_final_loss(
        args.input_root,
        split_percentile=args.split_percentile,
    )
    split_percent = int(args.split_percentile * 100)
    split_output_path = _split_output_path_for_metric(args.output)
    _plot_split_series(
        lower_series=lower_split_series,
        upper_series=upper_split_series,
        output_path=split_output_path,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        title="grad_norm_raw Split by Final VLM Loss",
        xlabel="denoise_step",
        ylabel="grad_norm_raw",
        lower_label=f"mean grad_norm_raw, {split_percent}% best final VLM losses",
        upper_label=f"mean grad_norm_raw, {split_percent}% worst final VLM losses",
        log_scale=True,
        vertical_lines=guidance_block_starts,
        x_tick_positions=guidance_tick_positions,
        x_tick_labels=guidance_tick_labels,
    )
    print(
        "Plotted grad_norm_raw split by final VLM loss: "
        f"{len(lower_split_series)} best-loss runs vs {len(upper_split_series)} worst-loss runs"
    )
    print(f"Saved plot to {split_output_path}")


if __name__ == "__main__":
    main()
# python metrics/statistics.py --split-percentile 0.2
