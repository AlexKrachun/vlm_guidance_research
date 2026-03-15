import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"pipeline", "prompt_index", "alignment_score", "quality_score"}
METRICS = ("alignment_score", "quality_score")
SCORE_VALUES = np.arange(20, 101, 20)
SCORE_SCALE = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize alignment and quality score distributions for all pipelines listed in a CSV file."
    )
    parser.add_argument(
        "--input",
        default="metrics/alignment_score_result.csv",
        help="Path to the alignment score CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/alignment_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=32,
        help="Number of bins for continuous-looking density visualizations.",
    )
    return parser.parse_args()


def _import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _load_rows(csv_path: Path) -> list[dict[str, str | int | float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: list[dict[str, str | int | float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"CSV file is missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "pipeline": row["pipeline"].strip(),
                    "prompt_index": int(row["prompt_index"]),
                    "alignment_score": float(row["alignment_score"]) * SCORE_SCALE,
                    "quality_score": float(row["quality_score"]) * SCORE_SCALE,
                }
            )

    if not rows:
        raise ValueError(f"No scores found in {csv_path}")

    return rows


def _group_metric(rows: list[dict[str, str | int | float]], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        pipeline = str(row["pipeline"])
        grouped.setdefault(pipeline, []).append(float(row[metric]))
    return dict(sorted(grouped.items()))


def _pivot_metric(rows: list[dict[str, str | int | float]], metric: str) -> tuple[list[int], list[str], np.ndarray]:
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    prompt_indices = sorted({int(row["prompt_index"]) for row in rows})
    pipeline_to_col = {pipeline: index for index, pipeline in enumerate(pipelines)}
    prompt_to_row = {prompt_index: index for index, prompt_index in enumerate(prompt_indices)}

    matrix = np.full((len(prompt_indices), len(pipelines)), np.nan, dtype=float)
    for row in rows:
        matrix[prompt_to_row[int(row["prompt_index"])]][pipeline_to_col[str(row["pipeline"])] ] = float(row[metric])

    return prompt_indices, pipelines, matrix


def _pivot_alignment_quality(rows: list[dict[str, str | int | float]]) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    alignment: dict[str, list[float]] = {pipeline: [] for pipeline in pipelines}
    quality: dict[str, list[float]] = {pipeline: [] for pipeline in pipelines}

    for row in rows:
        pipeline = str(row["pipeline"])
        alignment[pipeline].append(float(row["alignment_score"]))
        quality[pipeline].append(float(row["quality_score"]))

    return pipelines, {k: np.asarray(v, dtype=float) for k, v in alignment.items()}, {k: np.asarray(v, dtype=float) for k, v in quality.items()}


def _get_colors(plt, pipelines: list[str]) -> dict[str, tuple[float, ...]]:
    cmap = plt.get_cmap("tab10", len(pipelines))
    return {pipeline: cmap(index) for index, pipeline in enumerate(pipelines)}


def _style_axes(ax) -> None:
    ax.set_facecolor("#fffaf2")
    ax.grid(True, axis="y", alpha=0.22, color="#7f8c69", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5f6f52")
    ax.spines["bottom"].set_color("#5f6f52")


def _metric_title(metric: str) -> str:
    return metric.replace("_", " ").title()


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _score_distribution(values: np.ndarray) -> np.ndarray:
    counts = np.array([(values == score).sum() for score in SCORE_VALUES], dtype=float)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, y


def _smooth_curve(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=(10.0, 110.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def _save_violin_plot(plt, grouped_scores: dict[str, list[float]], metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    pipelines = list(grouped_scores.keys())
    score_lists = [grouped_scores[pipeline] for pipeline in pipelines]

    fig, ax = plt.subplots(figsize=(max(10, len(pipelines) * 1.6), 7), facecolor="#f7f3eb")
    violin = ax.violinplot(score_lists, showmeans=True, showmedians=True, widths=0.85)

    for body, pipeline in zip(violin["bodies"], pipelines):
        body.set_facecolor(colors[pipeline])
        body.set_edgecolor(colors[pipeline])
        body.set_alpha(0.45)
    violin["cmeans"].set_color("#bc6c25")
    violin["cmeans"].set_linewidth(2.0)
    violin["cmedians"].set_color("#283618")
    violin["cmedians"].set_linewidth(2.0)

    for index, pipeline in enumerate(pipelines, start=1):
        scores = np.asarray(grouped_scores[pipeline], dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(scores)) if len(scores) > 1 else np.array([0.0])
        ax.scatter(np.full(len(scores), index) + jitter, scores, s=14, alpha=0.18, color=colors[pipeline], edgecolors="none")

    ax.set_xticks(np.arange(1, len(pipelines) + 1))
    ax.set_xticklabels(pipelines, rotation=20, ha="right")
    ax.set_ylim(15.0, 105.0)
    ax.set_yticks(SCORE_VALUES)
    ax.set_title(f"{_metric_title(metric)}", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel(_metric_title(metric))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_violin.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_grouped_histogram(plt, grouped_scores: dict[str, list[float]], metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    pipelines = list(grouped_scores.keys())
    x = np.arange(len(SCORE_VALUES))
    width = 0.8 / max(1, len(pipelines))

    fig, ax = plt.subplots(figsize=(max(10, len(pipelines) * 1.8), 7), facecolor="#f7f3eb")
    for index, pipeline in enumerate(pipelines):
        distribution = _score_distribution(np.asarray(grouped_scores[pipeline], dtype=float))
        offset = (index - (len(pipelines) - 1) / 2.0) * width
        ax.bar(x + offset, distribution, width=width * 0.92, color=colors[pipeline], alpha=0.82, label=pipeline)

    ax.set_xticks(x)
    ax.set_xticklabels([str(score) for score in SCORE_VALUES])
    ax.set_title(f"{_metric_title(metric)} Grouped Histogram", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Score")
    ax.set_ylabel("Fraction of images")
    ax.legend(frameon=False, ncol=min(3, len(pipelines)))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_grouped_histogram.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_ecdf_plot(plt, grouped_scores: dict[str, list[float]], metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#f7f3eb")
    for pipeline, scores in grouped_scores.items():
        x, y = _ecdf(np.asarray(scores, dtype=float))
        ax.step(x, y, where="post", linewidth=2.4, color=colors[pipeline], label=f"{pipeline} (n={len(scores)})")

    ax.set_xticks(SCORE_VALUES)
    ax.set_xlim(18.0, 102.0)
    ax.set_title(f"{_metric_title(metric)} ECDF", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel(_metric_title(metric))
    ax.set_ylabel("Fraction of images")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_ecdf.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(plt, grouped_scores: dict[str, list[float]], metric: str, output_dir: Path) -> None:
    pipelines = list(grouped_scores.keys())
    matrix = np.vstack([_score_distribution(np.asarray(grouped_scores[pipeline], dtype=float)) for pipeline in pipelines])

    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.4), 6), facecolor="#f7f3eb")
    image = ax.imshow(matrix.T, aspect="auto", cmap="YlOrBr", interpolation="nearest", origin="lower")

    ax.set_title(f"{_metric_title(metric)} Heatmap", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Score")
    ax.set_xticks(np.arange(len(pipelines)))
    ax.set_xticklabels(pipelines, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCORE_VALUES)))
    ax.set_yticklabels([str(score) for score in SCORE_VALUES])

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Fraction of images")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_per_prompt_line_plot(plt, prompt_indices: list[int], pipelines: list[str], matrix: np.ndarray, metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#f7f3eb")
    for column, pipeline in enumerate(pipelines):
        scores = matrix[:, column]
        ax.plot(prompt_indices, scores, color=colors[pipeline], linewidth=1.8, alpha=0.9, label=pipeline)
        ax.scatter(prompt_indices, scores, color=colors[pipeline], s=14, alpha=0.5)

    ax.set_title(f"{_metric_title(metric)} per Prompt", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Prompt index")
    ax.set_ylabel(_metric_title(metric))
    ax.set_ylim(15.0, 105.0)
    ax.set_yticks(SCORE_VALUES)
    ax.legend(frameon=False, ncol=min(3, len(pipelines)))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_per_prompt_lines.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_pairwise_difference_plots(plt, prompt_indices: list[int], pipelines: list[str], matrix: np.ndarray, metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    for left_name, right_name in combinations(pipelines, 2):
        left_scores = matrix[:, pipelines.index(left_name)]
        right_scores = matrix[:, pipelines.index(right_name)]
        diff = left_scores - right_scores

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f7f3eb")
        positive_mask = diff >= 0
        negative_mask = ~positive_mask

        ax.axhline(0.0, color="#283618", linewidth=1.6, linestyle="--")
        ax.plot(prompt_indices, diff, color="#606c38", linewidth=1.8, alpha=0.9)
        ax.scatter(np.asarray(prompt_indices)[positive_mask], diff[positive_mask], color=colors[left_name], s=26, alpha=0.75, label=f"{left_name} >= {right_name}")
        ax.scatter(np.asarray(prompt_indices)[negative_mask], diff[negative_mask], color=colors[right_name], s=26, alpha=0.75, label=f"{right_name} > {left_name}")
        ax.fill_between(prompt_indices, 0.0, diff, where=positive_mask, color=colors[left_name], alpha=0.18)
        ax.fill_between(prompt_indices, 0.0, diff, where=negative_mask, color=colors[right_name], alpha=0.18)

        mean_diff = float(np.nanmean(diff))
        ax.text(
            0.01,
            0.97,
            f"mean diff = {mean_diff:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#283618",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffaf2", "edgecolor": "#bc6c25", "alpha": 0.95},
        )

        ax.set_title(f"{_metric_title(metric)} Difference: {left_name} - {right_name}", fontsize=17, fontweight="bold", color="#283618")
        ax.set_xlabel("Prompt index")
        ax.set_ylabel(f"{_metric_title(metric)} difference")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        safe_name = f"{metric}_diff_{_safe_name(left_name)}_vs_{_safe_name(right_name)}"
        fig.savefig(output_dir / f"{safe_name}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _save_scatter_alignment_quality(plt, pipelines: list[str], alignment: dict[str, np.ndarray], quality: dict[str, np.ndarray], output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#f7f3eb")
    for pipeline in pipelines:
        x = alignment[pipeline]
        y = quality[pipeline]
        jitter_x = np.linspace(-0.08, 0.08, len(x)) if len(x) > 1 else np.array([0.0])
        jitter_y = np.linspace(0.08, -0.08, len(y)) if len(y) > 1 else np.array([0.0])
        ax.scatter(x + jitter_x, y + jitter_y, s=32, alpha=0.45, color=colors[pipeline], label=pipeline)

    ax.set_xlim(15.0, 105.0)
    ax.set_ylim(15.0, 105.0)
    ax.set_xticks(SCORE_VALUES)
    ax.set_yticks(SCORE_VALUES)
    ax.set_title("Alignment vs Quality Scatter", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Alignment score")
    ax.set_ylabel("Quality score")
    ax.legend(frameon=False, ncol=min(3, len(pipelines)))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_quality_scatter.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_hexbin_alignment_quality(plt, pipelines: list[str], alignment: dict[str, np.ndarray], quality: dict[str, np.ndarray], output_dir: Path) -> None:
    columns = min(3, len(pipelines))
    rows = int(np.ceil(len(pipelines) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5.0 * rows), facecolor="#f7f3eb")
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    for index, pipeline in enumerate(pipelines):
        ax = axes_array[index // columns, index % columns]
        hexbin_artist = ax.hexbin(
            alignment[pipeline],
            quality[pipeline],
            gridsize=5,
            extent=(10.0, 110.0, 10.0, 110.0),
            mincnt=1,
            cmap="YlGnBu",
        )
        ax.set_title(pipeline, fontsize=13, fontweight="bold", color="#283618")
        ax.set_xlim(15.0, 105.0)
        ax.set_ylim(15.0, 105.0)
        ax.set_xticks(SCORE_VALUES)
        ax.set_yticks(SCORE_VALUES)
        ax.set_xlabel("Alignment")
        ax.set_ylabel("Quality")
        _style_axes(ax)
        fig.colorbar(hexbin_artist, ax=ax, shrink=0.85, label="Count")

    for index in range(len(pipelines), rows * columns):
        axes_array[index // columns, index % columns].axis("off")

    fig.suptitle("Alignment vs Quality 2D Density / Hexbin", fontsize=18, fontweight="bold", color="#283618")
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_quality_hexbin.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_ridgeline_plot(plt, grouped_scores: dict[str, list[float]], metric: str, output_dir: Path, colors: dict[str, tuple[float, ...]], bins: int) -> None:
    pipelines = list(grouped_scores.keys())
    fig, ax = plt.subplots(figsize=(11, max(6, len(pipelines) * 1.2)), facecolor="#f7f3eb")

    for index, pipeline in enumerate(pipelines):
        values = np.asarray(grouped_scores[pipeline], dtype=float)
        x, smooth = _smooth_curve(values, bins=bins)
        baseline = index * 1.0
        ax.fill_between(x, baseline, baseline + smooth, color=colors[pipeline], alpha=0.45)
        ax.plot(x, baseline + smooth, color=colors[pipeline], linewidth=2.0)
        ax.hlines(baseline, xmin=20.0, xmax=100.0, color="#d8d1bf", linewidth=0.8)

    ax.set_yticks(np.arange(len(pipelines)))
    ax.set_yticklabels(pipelines)
    ax.set_xlim(20.0, 100.0)
    ax.set_xticks(SCORE_VALUES)
    ax.set_title(f"{_metric_title(metric)} Ridgeline Plot", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel(_metric_title(metric))
    ax.set_ylabel("Pipeline")
    ax.grid(True, axis="x", alpha=0.18, color="#7f8c69", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#5f6f52")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_ridgeline.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_gap_plot(plt, pipelines: list[str], alignment: dict[str, np.ndarray], quality: dict[str, np.ndarray], output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    gap_values = np.arange(-80, 81, 20)
    x = np.arange(len(gap_values))
    width = 0.8 / max(1, len(pipelines))

    fig, ax = plt.subplots(figsize=(max(10, len(pipelines) * 1.8), 7), facecolor="#f7f3eb")
    for index, pipeline in enumerate(pipelines):
        gaps = alignment[pipeline] - quality[pipeline]
        counts = np.array([(gaps == gap).sum() for gap in gap_values], dtype=float)
        distribution = counts / counts.sum() if counts.sum() else counts
        offset = (index - (len(pipelines) - 1) / 2.0) * width
        ax.bar(x + offset, distribution, width=width * 0.92, color=colors[pipeline], alpha=0.82, label=pipeline)

    ax.axvline(np.where(gap_values == 0)[0][0], color="#283618", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in gap_values])
    ax.set_title("Alignment - Quality Gap Plot", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Alignment score minus quality score")
    ax.set_ylabel("Fraction of images")
    ax.legend(frameon=False, ncol=min(3, len(pipelines)))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_quality_gap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_summary_text(rows: list[dict[str, str | int | float]]) -> str:
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    lines = ["Alignment/quality summary:"]
    for pipeline in pipelines:
        pipeline_rows = [row for row in rows if str(row["pipeline"]) == pipeline]
        alignment_scores = np.asarray([float(row["alignment_score"]) for row in pipeline_rows], dtype=float)
        quality_scores = np.asarray([float(row["quality_score"]) for row in pipeline_rows], dtype=float)
        lines.append(
            f"- {pipeline}: n={len(pipeline_rows)}, alignment_mean={alignment_scores.mean():.3f}, "
            f"quality_mean={quality_scores.mean():.3f}, gap_mean={(alignment_scores - quality_scores).mean():.3f}"
        )
    return "\n".join(lines)


def _write_summary_file(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    summary_text = _build_summary_text(rows)
    summary_path = output_dir / "alignment_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    print(f"Saved summary to {summary_path}")


def main() -> None:
    args = _parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be a positive integer")

    plt = _import_matplotlib()
    rows = _load_rows(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipelines, alignment_arrays, quality_arrays = _pivot_alignment_quality(rows)
    colors = _get_colors(plt, pipelines)

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

    for metric in METRICS:
        grouped_scores = _group_metric(rows, metric)
        prompt_indices, metric_pipelines, matrix = _pivot_metric(rows, metric)
        _save_violin_plot(plt, grouped_scores, metric, output_dir, colors)
        _save_grouped_histogram(plt, grouped_scores, metric, output_dir, colors)
        _save_ecdf_plot(plt, grouped_scores, metric, output_dir, colors)
        _save_heatmap(plt, grouped_scores, metric, output_dir)
        _save_per_prompt_line_plot(plt, prompt_indices, metric_pipelines, matrix, metric, output_dir, colors)
        _save_pairwise_difference_plots(plt, prompt_indices, metric_pipelines, matrix, metric, output_dir, colors)
        _save_ridgeline_plot(plt, grouped_scores, metric, output_dir, colors, args.bins)

    _save_scatter_alignment_quality(plt, pipelines, alignment_arrays, quality_arrays, output_dir, colors)
    _save_hexbin_alignment_quality(plt, pipelines, alignment_arrays, quality_arrays, output_dir)
    _save_gap_plot(plt, pipelines, alignment_arrays, quality_arrays, output_dir, colors)
    _write_summary_file(output_dir, rows)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
