import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {"pipeline", "clip_score", "prompt_index"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CLIP score distributions for all pipelines listed in a CSV file."
    )
    parser.add_argument(
        "--input",
        default="metrics/clip_score_result.csv",
        help="Path to the CLIP score CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/clip_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=28,
        help="Number of histogram bins for derived density plots.",
    )
    return parser.parse_args()


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
                    "clip_score": float(row["clip_score"]),
                }
            )

    if not rows:
        raise ValueError(f"No CLIP scores found in {csv_path}")

    return rows


def _group_scores(rows: list[dict[str, str | int | float]]) -> dict[str, list[float]]:
    grouped_scores: dict[str, list[float]] = {}
    for row in rows:
        pipeline = str(row["pipeline"])
        grouped_scores.setdefault(pipeline, []).append(float(row["clip_score"]))
    return dict(sorted(grouped_scores.items()))


def _pivot_scores(rows: list[dict[str, str | int | float]]) -> tuple[list[int], list[str], np.ndarray]:
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    prompt_indices = sorted({int(row["prompt_index"]) for row in rows})
    pipeline_to_col = {pipeline: index for index, pipeline in enumerate(pipelines)}
    prompt_to_row = {prompt_index: index for index, prompt_index in enumerate(prompt_indices)}

    matrix = np.full((len(prompt_indices), len(pipelines)), np.nan, dtype=float)
    for row in rows:
        matrix[prompt_to_row[int(row["prompt_index"])]][pipeline_to_col[str(row["pipeline"])] ] = float(row["clip_score"])

    return prompt_indices, pipelines, matrix


def _get_colors(pipelines: list[str]) -> dict[str, tuple[float, ...]]:
    cmap = plt.get_cmap("tab10", len(pipelines))
    return {pipeline: cmap(index) for index, pipeline in enumerate(pipelines)}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#fffaf2")
    ax.grid(True, axis="y", alpha=0.22, color="#7f8c69", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5f6f52")
    ax.spines["bottom"].set_color("#5f6f52")


def _save_violin_plot(grouped_scores: dict[str, list[float]], output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
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
    ax.set_title("CLIP Score", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("CLIP score")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "clip_score_violin.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, y


def _save_ecdf_plot(grouped_scores: dict[str, list[float]], output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#f7f3eb")
    for pipeline, scores in grouped_scores.items():
        x, y = _ecdf(np.asarray(scores, dtype=float))
        ax.step(x, y, where="post", linewidth=2.4, color=colors[pipeline], label=f"{pipeline} (n={len(scores)})")

    ax.set_title("CLIP Score ECDF", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("CLIP score")
    ax.set_ylabel("Fraction of prompts")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "clip_score_ecdf.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_per_prompt_line_plot(prompt_indices: list[int], pipelines: list[str], matrix: np.ndarray, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#f7f3eb")
    for column, pipeline in enumerate(pipelines):
        scores = matrix[:, column]
        ax.plot(prompt_indices, scores, color=colors[pipeline], linewidth=2.0, alpha=0.9, label=pipeline)
        ax.scatter(prompt_indices, scores, color=colors[pipeline], s=16, alpha=0.55)

    ax.set_title("CLIP Score per Prompt", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("CLIP score")
    ax.legend(frameon=False, ncol=min(3, len(pipelines)))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "clip_score_per_prompt_lines.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(prompt_indices: list[int], pipelines: list[str], matrix: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.4), max(8, len(prompt_indices) * 0.12)), facecolor="#f7f3eb")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap="YlOrBr", interpolation="nearest")

    ax.set_title("CLIP Score Heatmap", fontsize=18, fontweight="bold", color="#283618")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Prompt index")
    ax.set_xticks(np.arange(len(pipelines)))
    ax.set_xticklabels(pipelines, rotation=25, ha="right")

    if len(prompt_indices) <= 30:
        ax.set_yticks(np.arange(len(prompt_indices)))
        ax.set_yticklabels(prompt_indices)
    else:
        tick_positions = np.linspace(0, len(prompt_indices) - 1, num=10, dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([prompt_indices[pos] for pos in tick_positions])

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("CLIP score")
    fig.tight_layout()
    fig.savefig(output_dir / "clip_score_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_pairwise_difference_plots(prompt_indices: list[int], pipelines: list[str], matrix: np.ndarray, output_dir: Path, colors: dict[str, tuple[float, ...]]) -> None:
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
            f"mean diff = {mean_diff:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#283618",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffaf2", "edgecolor": "#bc6c25", "alpha": 0.95},
        )

        ax.set_title(f"Pairwise Difference: {left_name} - {right_name}", fontsize=17, fontweight="bold", color="#283618")
        ax.set_xlabel("Prompt index")
        ax.set_ylabel("CLIP score difference")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        safe_name = f"clip_score_diff_{left_name}_vs_{right_name}".replace("/", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _summary_lines(grouped_scores: dict[str, list[float]]) -> list[str]:
    lines = ["CLIP score summary:"]
    for pipeline, scores in sorted(grouped_scores.items(), key=lambda item: np.mean(item[1]), reverse=True):
        scores_array = np.asarray(scores, dtype=float)
        lines.append(
            f"- {pipeline}: n={len(scores)}, mean={np.mean(scores_array):.4f}, "
            f"median={np.median(scores_array):.4f}, std={np.std(scores_array):.4f}, "
            f"min={np.min(scores_array):.4f}, max={np.max(scores_array):.4f}"
        )
    return lines


def _save_summary_text(grouped_scores: dict[str, list[float]], output_dir: Path) -> Path:
    summary_path = output_dir / "clip_score_summary.txt"
    summary_path.write_text("\n".join(_summary_lines(grouped_scores)) + "\n", encoding="utf-8")
    return summary_path


def _print_summary(grouped_scores: dict[str, list[float]]) -> None:
    for line in _summary_lines(grouped_scores):
        print(line)


def main() -> None:
    args = _parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be a positive integer")

    rows = _load_rows(Path(args.input))
    grouped_scores = _group_scores(rows)
    prompt_indices, pipelines, matrix = _pivot_scores(rows)
    colors = _get_colors(pipelines)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    _save_violin_plot(grouped_scores, output_dir, colors)
    _save_ecdf_plot(grouped_scores, output_dir, colors)
    _save_per_prompt_line_plot(prompt_indices, pipelines, matrix, output_dir, colors)
    _save_heatmap(prompt_indices, pipelines, matrix, output_dir)
    _save_pairwise_difference_plots(prompt_indices, pipelines, matrix, output_dir, colors)
    summary_path = _save_summary_text(grouped_scores, output_dir)
    _print_summary(grouped_scores)
    print(f"Saved plots to {output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
