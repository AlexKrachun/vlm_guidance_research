import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import httpx
from tqdm import tqdm

from alignment import (
    DEFAULT_MODEL,
    OPENAI_API_URL,
    build_eval_prompt,
    encode_image,
    parse_evaluation_text,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PREFERRED_IMAGE_NAMES = ("img.png", "img_00.png", "image.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute alignment scores for generated images grouped by prompt and pipeline."
    )
    parser.add_argument(
        "-generations",
        "--generations",
        required=True,
        help="Path to the directory with prompt folders and pipeline subfolders.",
    )
    parser.add_argument(
        "--output",
        default="metrics/alignment_score_result.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vision-capable OpenAI model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the evaluator response.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of simultaneous OpenAI API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds for each API call.",
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


def _build_jobs(generations_dir: Path) -> list[dict[str, str | int]]:
    prompt_dirs = list(_iter_prompt_dirs(generations_dir))
    jobs: list[dict[str, str | int]] = []

    for fallback_index, prompt_dir in enumerate(prompt_dirs):
        prompt_index = _prompt_index(prompt_dir, fallback_index)
        for pipeline_dir in _iter_pipeline_dirs(prompt_dir):
            prompt_path = pipeline_dir / "prompt.txt"
            if not prompt_path.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            image_path = _find_image_file(pipeline_dir)
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            jobs.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_dir": prompt_dir.name,
                    "pipeline": pipeline_dir.name,
                    "image_path": str(image_path),
                    "prompt_path": str(prompt_path),
                    "prompt": prompt,
                }
            )

    return jobs


async def _evaluate_job(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    job: dict[str, str | int],
    api_key: str,
    model: str,
    max_tokens: int,
) -> dict[str, str | int]:
    async with semaphore:
        image_path = str(job["image_path"])
        prompt = str(job["prompt"])
        base64_image = encode_image(image_path)
        eval_prompt = build_eval_prompt(prompt)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": eval_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        response = await client.post(
            OPENAI_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text
            raise RuntimeError(
                f"OpenAI API request failed for {image_path}: {exc.response.status_code} {error_body}"
            ) from exc

        response_data = response.json()
        try:
            text = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI API response for {image_path}:\n{json.dumps(response_data, indent=2)}"
            ) from exc

        result = parse_evaluation_text(text)
        return {
            **job,
            "alignment_score": result["alignment score"],
            "alignment_explanation": result["alignment explanation"],
            "quality_score": result["quality score"],
            "quality_explanation": result["quality explanation"],
        }


async def _run_async(args: argparse.Namespace) -> list[dict[str, str | int]]:
    generations_dir = Path(args.generations)
    jobs = _build_jobs(generations_dir)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(
                _evaluate_job(
                    client=client,
                    semaphore=semaphore,
                    job=job,
                    api_key=args.api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
            )
            for job in jobs
        ]

        rows: list[dict[str, str | int]] = []
        with tqdm(total=len(tasks), desc="Alignment score requests") as progress:
            for future in asyncio.as_completed(tasks):
                rows.append(await future)
                progress.update(1)

    rows.sort(key=lambda row: (int(row["prompt_index"]), str(row["pipeline"])))
    return rows


def _write_csv(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "prompt_index",
                "prompt_dir",
                "pipeline",
                "alignment_score",
                "alignment_explanation",
                "quality_score",
                "quality_explanation",
                "image_path",
                "prompt_path",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    generations_dir = Path(args.generations)
    output_path = Path(args.output)

    if not generations_dir.is_dir():
        raise NotADirectoryError(f"Generations directory not found: {generations_dir}")
    if not args.api_key:
        raise SystemExit("OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive.")

    rows = asyncio.run(_run_async(args))
    _write_csv(output_path, rows)
    print(f"Saved {len(rows)} alignment scores to {output_path}")


if __name__ == "__main__":
    main()
