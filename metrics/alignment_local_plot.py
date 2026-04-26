import argparse
import csv
import fnmatch
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = {
    "pipeline_key",
    "experiment",
    "pipeline",
    "n",
    "alignment_mean",
    "quality_mean",
}

DETAIL_COLUMNS = {
    "pipeline_key",
    "experiment",
    "pipeline",
    "alignment_score",
    "quality_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quality-vs-alignment scatter from alignment_local.py CSV outputs."
    )
    parser.add_argument(
        "--input",
        default="metrics/alignment_local_whoops50bench_summary.csv",
        help="Summary or detailed CSV produced by metrics/alignment_local.py.",
    )
    parser.add_argument(
        "--output",
        default="metrics/alignment_local_whoops50bench_scatter.png",
        help="Output plot path. Extension controls format, e.g. .png, .pdf, .svg.",
    )
    parser.add_argument(
        "--title",
        default="Средние заначения Alignment и Quality для сочетаний параметров генерации",
        help="Plot title.",
    )
    parser.add_argument(
        "--label-column",
        default="pipeline_key",
        choices=("pipeline_key", "experiment", "pipeline"),
        help="Column to use as point labels.",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=0.18,
        help="Padding added around min/max scores when xlim/ylim are not provided.",
    )
    parser.add_argument("--figsize", nargs=2, type=float, default=(11.0, 7.8))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-labels", action="store_true", help="Do not annotate points.")
    parser.add_argument("--no-legend", action="store_true", help="Do not show the pipeline color legend.")
    parser.add_argument(
        "--point-size",
        type=float,
        default=110.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--label-adjust-iterations",
        type=int,
        default=120,
        help="Maximum label collision adjustment iterations.",
    )
    parser.add_argument(
        "--label-padding-px",
        type=float,
        default=2.0,
        help="Minimum screen-space padding between adjusted labels.",
    )
    parser.add_argument(
        "--section-column",
        default="experiment",
        choices=("pipeline_key", "experiment", "pipeline"),
        help="Point column matched by --section selectors.",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        metavar="NAME=SELECTOR[,SELECTOR...]",
        help=(
            "Define a named section. Selectors match --section-column values exactly, "
            "with shell wildcards, or as regex when prefixed with 're:'. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="NAME=SECTION[,SECTION...]",
        help=(
            "Save an additional plot containing only points from the listed sections. "
            "The file is saved next to --output with NAME appended to the stem. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--grouped-plot-column",
        default="experiment",
        choices=("pipeline_key", "experiment", "pipeline"),
        help="Point column matched by --grouped-plot selectors.",
    )
    parser.add_argument(
        "--grouped-plot",
        action="append",
        default=[],
        metavar="NAME=GROUP=SELECTOR[,SELECTOR...];GROUP=SELECTOR[,SELECTOR...]",
        help=(
            "Save an additional plot colored by ad-hoc groups. Selectors match "
            "--grouped-plot-column exactly, with shell wildcards, or as regex when "
            "prefixed with 're:'. Quote the argument in shell because groups are "
            "separated with semicolons. Can be provided multiple times."
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def load_points(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    columns = set(rows[0].keys())

    if SUMMARY_COLUMNS.issubset(columns):
        return points_from_summary(rows)
    if DETAIL_COLUMNS.issubset(columns):
        return points_from_detail(rows)

    raise ValueError(
        "Input CSV is neither a summary nor a detailed alignment_local.py CSV. "
        f"Columns found: {sorted(columns)}"
    )


def points_from_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        points.append(
            {
                "pipeline_key": row["pipeline_key"],
                "experiment": row["experiment"],
                "pipeline": row["pipeline"],
                "n": int(float(row["n"])),
                "quality_mean": float(row["quality_mean"]),
                "alignment_mean": float(row["alignment_mean"]),
                "quality_std": float(row.get("quality_std") or 0.0),
                "alignment_std": float(row.get("alignment_std") or 0.0),
            }
        )
    return sorted(points, key=lambda point: str(point["pipeline_key"]))


def points_from_detail(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row.get("alignment_score") or not row.get("quality_score"):
            continue
        grouped[row["pipeline_key"]].append(row)

    points = []
    for pipeline_key, group in grouped.items():
        alignment = [float(row["alignment_score"]) for row in group]
        quality = [float(row["quality_score"]) for row in group]
        first = group[0]
        points.append(
            {
                "pipeline_key": pipeline_key,
                "experiment": first["experiment"],
                "pipeline": first["pipeline"],
                "n": len(group),
                "quality_mean": mean(quality),
                "alignment_mean": mean(alignment),
                "quality_std": std(quality),
                "alignment_std": std(alignment),
            }
        )
    if not points:
        raise ValueError("Detailed CSV has no completed score rows.")
    return sorted(points, key=lambda point: str(point["pipeline_key"]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def save_plot(
    points: list[dict[str, Any]],
    args: argparse.Namespace,
    output_path: Path,
    title: str,
    color_column: str,
    colors: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to plot: pip install matplotlib") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    annotations = []
    for point in points:
        x = float(point["quality_mean"])
        y = float(point["alignment_mean"])
        color_value = str(point[color_column])
        ax.scatter(
            x,
            y,
            s=args.point_size,
            color=colors[color_value],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
            label=color_value,
        )
        if not args.no_labels:
            label = point_label(point, args.label_column)
            annotations.append(
                ax.annotate(label, (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8)
            )

    ax.set_xlabel("Quality score")
    ax.set_ylabel("Alignment score")
    ax.set_xlim(*resolve_axis_limits([float(point["quality_mean"]) for point in points], args.xlim, args.axis_padding))
    ax.set_ylim(*resolve_axis_limits([float(point["alignment_mean"]) for point in points], args.ylim, args.axis_padding))
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.set_facecolor("#fafafa")

    if not args.no_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = {}
        for handle, label in zip(handles, labels):
            unique_handles.setdefault(label, handle)
        ax.legend(
            unique_handles.values(),
            unique_handles.keys(),
            title=color_column.replace("_", " ").title(),
            frameon=True,
            framealpha=0.92,
            loc="best",
        )

    fig.tight_layout()
    adjust_label_positions(fig, ax, annotations, args.label_adjust_iterations, args.label_padding_px)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def build_colors(points: list[dict[str, Any]], plt: Any, color_column: str) -> dict[str, Any]:
    values = sorted({str(point[color_column]) for point in points})
    palette_name = "tab10" if len(values) <= 10 else "tab20"
    colors = plt.get_cmap(palette_name).colors
    return {value: colors[index % len(colors)] for index, value in enumerate(values)}


def point_label(point: dict[str, Any], label_column: str) -> str:
    return str(point[label_column]).split("/", 1)[0]


def adjust_label_positions(
    fig: Any,
    ax: Any,
    annotations: list[Any],
    max_iterations: int,
    padding_px: float,
) -> None:
    if len(annotations) < 2 or max_iterations <= 0:
        return

    from matplotlib.transforms import Bbox

    px_to_points = 72.0 / fig.dpi
    for _ in range(max_iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [padded_bbox(annotation.get_window_extent(renderer), padding_px) for annotation in annotations]
        deltas = [[0.0, 0.0] for _ in annotations]
        moved = False

        for left_index in range(len(boxes)):
            for right_index in range(left_index + 1, len(boxes)):
                left_box = boxes[left_index]
                right_box = boxes[right_index]
                overlap_x = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
                overlap_y = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                left_center_x = (left_box.x0 + left_box.x1) / 2
                right_center_x = (right_box.x0 + right_box.x1) / 2
                left_center_y = (left_box.y0 + left_box.y1) / 2
                right_center_y = (right_box.y0 + right_box.y1) / 2

                if overlap_x < overlap_y:
                    direction = -1.0 if left_center_x <= right_center_x else 1.0
                    shift = min(overlap_x / 2 + padding_px, 10.0)
                    deltas[left_index][0] += direction * shift
                    deltas[right_index][0] -= direction * shift
                else:
                    direction = -1.0 if left_center_y <= right_center_y else 1.0
                    shift = min(overlap_y / 2 + padding_px, 10.0)
                    deltas[left_index][1] += direction * shift
                    deltas[right_index][1] -= direction * shift
                moved = True

        axes_box = ax.get_window_extent(renderer)
        for index, box in enumerate(boxes):
            next_box = Bbox.from_extents(
                box.x0 + deltas[index][0],
                box.y0 + deltas[index][1],
                box.x1 + deltas[index][0],
                box.y1 + deltas[index][1],
            )
            if next_box.x0 < axes_box.x0:
                deltas[index][0] += axes_box.x0 - next_box.x0
                moved = True
            if next_box.x1 > axes_box.x1:
                deltas[index][0] -= next_box.x1 - axes_box.x1
                moved = True
            if next_box.y0 < axes_box.y0:
                deltas[index][1] += axes_box.y0 - next_box.y0
                moved = True
            if next_box.y1 > axes_box.y1:
                deltas[index][1] -= next_box.y1 - axes_box.y1
                moved = True

        if not moved:
            break

        for annotation, (dx_px, dy_px) in zip(annotations, deltas):
            if dx_px == 0 and dy_px == 0:
                continue
            x_points, y_points = annotation.get_position()
            annotation.set_position((x_points + dx_px * px_to_points, y_points + dy_px * px_to_points))


def padded_bbox(bbox: Any, padding_px: float) -> Any:
    return bbox.from_extents(
        bbox.x0 - padding_px,
        bbox.y0 - padding_px,
        bbox.x1 + padding_px,
        bbox.y1 + padding_px,
    )


def resolve_axis_limits(values: list[float], explicit_limits: list[float] | None, padding: float) -> tuple[float, float]:
    if explicit_limits is not None:
        return float(explicit_limits[0]), float(explicit_limits[1])

    value_min = min(values)
    value_max = max(values)
    if math.isclose(value_min, value_max):
        value_min -= padding
        value_max += padding
    else:
        value_min -= padding
        value_max += padding

    return max(0.0, value_min), min(5.5, value_max)


def parse_named_list_specs(specs: list[str], option_name: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{option_name} must look like NAME=value1,value2: {spec}")
        name, raw_values = spec.split("=", 1)
        name = name.strip()
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not name or not values:
            raise ValueError(f"{option_name} must include a non-empty name and values: {spec}")
        if name in parsed:
            raise ValueError(f"Duplicate {option_name} name: {name}")
        parsed[name] = values
    return parsed


def parse_grouped_plot_specs(specs: list[str]) -> dict[str, dict[str, list[str]]]:
    parsed: dict[str, dict[str, list[str]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--grouped-plot must look like NAME=GROUP=selector;GROUP=selector: {spec}")
        plot_name, raw_groups = spec.split("=", 1)
        plot_name = plot_name.strip()
        if not plot_name or not raw_groups.strip():
            raise ValueError(f"--grouped-plot must include a non-empty name and groups: {spec}")
        if plot_name in parsed:
            raise ValueError(f"Duplicate --grouped-plot name: {plot_name}")

        group_specs: dict[str, list[str]] = {}
        for raw_group in raw_groups.split(";"):
            raw_group = raw_group.strip()
            if not raw_group:
                continue
            if "=" not in raw_group:
                raise ValueError(f"--grouped-plot group must look like GROUP=selector1,selector2: {raw_group}")
            group_name, raw_selectors = raw_group.split("=", 1)
            group_name = group_name.strip()
            selectors = [selector.strip() for selector in raw_selectors.split(",") if selector.strip()]
            if not group_name or not selectors:
                raise ValueError(f"--grouped-plot group must include a non-empty name and selectors: {raw_group}")
            if group_name in group_specs:
                raise ValueError(f"Duplicate group '{group_name}' in --grouped-plot '{plot_name}'")
            group_specs[group_name] = selectors

        if not group_specs:
            raise ValueError(f"--grouped-plot must include at least one group: {spec}")
        parsed[plot_name] = group_specs
    return parsed


def assign_sections(
    points: list[dict[str, Any]],
    section_specs: dict[str, list[str]],
    section_column: str,
) -> dict[str, int]:
    section_counts = dict.fromkeys(section_specs, 0)
    for point in points:
        point["section"] = "unsectioned"
        value = str(point[section_column])
        for section_name, selectors in section_specs.items():
            if any(matches_selector(value, selector) for selector in selectors):
                point["section"] = section_name
                section_counts[section_name] += 1
                break
    return section_counts


def points_with_group(
    points: list[dict[str, Any]],
    group_specs: dict[str, list[str]],
    group_column: str,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    grouped_points = []
    group_counts = dict.fromkeys(group_specs, 0)
    unmatched_count = 0

    for point in points:
        value = str(point[group_column])
        matched_group = None
        for group_name, selectors in group_specs.items():
            if any(matches_selector(value, selector) for selector in selectors):
                matched_group = group_name
                break
        if matched_group is None:
            unmatched_count += 1
            continue

        grouped_point = dict(point)
        grouped_point["group"] = matched_group
        grouped_points.append(grouped_point)
        group_counts[matched_group] += 1

    return grouped_points, group_counts, unmatched_count


def matches_selector(value: str, selector: str) -> bool:
    if selector.startswith("re:"):
        return re.search(selector[3:], value) is not None
    if any(char in selector for char in "*?[]"):
        return fnmatch.fnmatchcase(value, selector)
    return value == selector


def points_for_sections(points: list[dict[str, Any]], sections: list[str]) -> list[dict[str, Any]]:
    requested = set(sections)
    return [point for point in points if str(point.get("section")) in requested]


def comparison_output_path(output_path: Path, comparison_name: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{slugify(comparison_name)}{output_path.suffix}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "comparison"


def main() -> None:
    args = parse_args()
    points = load_points(Path(args.input))
    output_path = Path(args.output)
    section_specs = parse_named_list_specs(args.section, "--section")
    comparison_specs = parse_named_list_specs(args.comparison, "--comparison")
    grouped_plot_specs = parse_grouped_plot_specs(args.grouped_plot)

    color_column = "pipeline"
    if section_specs:
        section_counts = assign_sections(points, section_specs, args.section_column)
        empty_sections = [name for name, count in section_counts.items() if count == 0]
        if empty_sections:
            raise ValueError(
                "Sections matched no points: "
                f"{', '.join(empty_sections)}. Check --section-column and selectors."
            )
        color_column = "section"

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to plot: pip install matplotlib") from exc
    colors = build_colors(points, plt, color_column)

    save_plot(points, args, output_path, args.title, color_column, colors)
    print(f"Saved scatter plot with {len(points)} points to {output_path}")

    if comparison_specs and not section_specs:
        raise ValueError("--comparison requires at least one --section.")

    for comparison_name, section_names in comparison_specs.items():
        missing_sections = [name for name in section_names if name not in section_specs]
        if missing_sections:
            raise ValueError(
                f"Comparison '{comparison_name}' references unknown sections: "
                f"{', '.join(missing_sections)}"
            )
        comparison_points = points_for_sections(points, section_names)
        if not comparison_points:
            raise ValueError(f"Comparison '{comparison_name}' matched no points.")
        comparison_path = comparison_output_path(output_path, comparison_name)
        comparison_title = f"{args.title}: {comparison_name}"
        save_plot(comparison_points, args, comparison_path, comparison_title, "section", colors)
        print(
            f"Saved comparison '{comparison_name}' with {len(comparison_points)} "
            f"points to {comparison_path}"
        )

    for plot_name, group_specs in grouped_plot_specs.items():
        grouped_points, group_counts, unmatched_count = points_with_group(
            points,
            group_specs,
            args.grouped_plot_column,
        )
        empty_groups = [name for name, count in group_counts.items() if count == 0]
        if empty_groups:
            raise ValueError(
                f"Grouped plot '{plot_name}' has groups that matched no points: "
                f"{', '.join(empty_groups)}"
            )
        if not grouped_points:
            raise ValueError(f"Grouped plot '{plot_name}' matched no points.")

        grouped_path = comparison_output_path(output_path, plot_name)
        grouped_title = f"{args.title}: {plot_name}"
        grouped_colors = build_colors(grouped_points, plt, "group")
        save_plot(grouped_points, args, grouped_path, grouped_title, "group", grouped_colors)
        message = f"Saved grouped plot '{plot_name}' with {len(grouped_points)} points to {grouped_path}"
        if unmatched_count:
            message += f" ({unmatched_count} unmatched points skipped)"
        print(message)


if __name__ == "__main__":
    main()
