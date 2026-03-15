import argparse
import csv
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from clip_score import CLIPScorePipeline


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy"}
PREFERRED_IMAGE_NAMES = ("img.png", "img_00.png", "image.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP scores for generated images grouped by prompt and pipeline."
    )
    parser.add_argument(
        "-generations",
        "--generations",
        required=True,
        help="Path to the directory with prompt folders and pipeline subfolders.",
    )
    parser.add_argument(
        "--output",
        default="metrics/clip_score_result.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model",
        default="openai:ViT-L-14",
        help="open_clip model id in PRETRAINED:ARCH format.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./hf_cache/",
        help="Directory for downloaded model weights.",
    )
    return parser.parse_args()


def _iter_prompt_dirs(generations_dir: Path) -> Iterable[Path]:
    return sorted(path for path in generations_dir.iterdir() if path.is_dir())


def _iter_pipeline_dirs(prompt_dir: Path) -> Iterable[Path]:
    return sorted(path for path in prompt_dir.iterdir() if path.is_dir())


def _find_image_file(pipeline_dir: Path) -> Path:
    for name in PREFERRED_IMAGE_NAMES:
        candidate = pipeline_dir / name
        if candidate.is_file():
            return candidate

    image_files = sorted(
        path for path in pipeline_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No image file found in {pipeline_dir}")
    if len(image_files) > 1:
        raise ValueError(
            f"Multiple image files found in {pipeline_dir}: {[path.name for path in image_files]}"
        )
    return image_files[0]


def _prompt_index(prompt_dir: Path, fallback_index: int) -> int:
    prefix = prompt_dir.name.split("_", maxsplit=1)[0]
    if prefix.isdigit():
        return int(prefix)
    return fallback_index


def main() -> None:
    args = _parse_args()
    generations_dir = Path(args.generations)
    output_path = Path(args.output)

    if not generations_dir.is_dir():
        raise NotADirectoryError(f"Generations directory not found: {generations_dir}")

    pipeline = CLIPScorePipeline(
        model_name=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    prompt_dirs = list(_iter_prompt_dirs(generations_dir))
    rows: list[dict[str, str | int | float]] = []
    for fallback_index, prompt_dir in enumerate(tqdm(prompt_dirs, desc="CLIP score prompts")):
        prompt_index = _prompt_index(prompt_dir, fallback_index)
        for pipeline_dir in _iter_pipeline_dirs(prompt_dir):
            prompt_path = pipeline_dir / "prompt.txt"
            if not prompt_path.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            image_path = _find_image_file(pipeline_dir)
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            clip_score = pipeline.score(image_path, prompt)

            rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_dir": prompt_dir.name,
                    "pipeline": pipeline_dir.name,
                    "clip_score": clip_score,
                    "image_path": str(image_path),
                    "prompt_path": str(prompt_path),
                    "prompt": prompt,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "prompt_index",
                "prompt_dir",
                "pipeline",
                "clip_score",
                "image_path",
                "prompt_path",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} CLIP scores to {output_path}")


if __name__ == "__main__":
    main()
