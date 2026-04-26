import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any) -> Any:
        return iterable


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PREFERRED_IMAGE_NAMES = ("img.png", "img_00.png", "image.png")

DETAIL_FIELDNAMES = [
    "experiment",
    "pipeline",
    "pipeline_key",
    "prompt_index",
    "prompt_dir",
    "prompt",
    "alignment_score",
    "alignment_explanation",
    "quality_score",
    "quality_explanation",
    "image_hash",
    "image_path",
    "prompt_path",
    "model",
    "raw_response",
]

SUMMARY_FIELDNAMES = [
    "pipeline_key",
    "experiment",
    "pipeline",
    "n",
    "alignment_mean",
    "alignment_std",
    "quality_mean",
    "quality_std",
]


def build_eval_prompt(prompt: str, explanation_max_words: int = 18) -> str:
    return f"""You are an assistant evaluating an image on two independent aspects:
(1) how well it aligns with the meaning of a given text prompt, and
(2) its visual quality.

The text prompt is: "{prompt}"

---

PART 1: PROMPT ALIGNMENT (Semantic Fidelity)
Evaluate only the meaning conveyed by the image - ignore visual artifacts.
Focus on:
- Are the correct objects present and depicted in a way that clearly demonstrates their intended roles and actions from the prompt?
- Does the scene illustrate the intended situation or use-case in a concrete and functional way, rather than through symbolic, metaphorical, or hybrid representation?
- If the described usage or interaction is missing or unclear, alignment should be penalized.
- Focus strictly on the presence, roles, and relationships of the described elements - not on rendering quality.

Score from 1 to 5:
5: Fully conveys the prompt's meaning with correct elements
4: Mostly accurate - main elements are correct, with minor conceptual or contextual issues
3: Main subjects are present but important attributes or actions are missing or wrong
2: Some relevant components are present, but key elements or intent are significantly misrepresented
1: Does not reflect the prompt at all

---

PART 2: VISUAL QUALITY (Rendering Fidelity)
Now focus only on how the image looks visually - ignore whether it matches the prompt.
Focus on:
- Are there rendering artifacts, distortions, or broken elements?
- Are complex areas like faces, hands, and shaped objects well-formed and visually coherent?
- Are complex areas like faces, hands, limbs, and object grips well-formed and anatomically correct?
- Is lighting, texture, and perspective consistent across the scene?
- Do elements appear physically coherent - i.e., do objects connect naturally (no floating tools, clipped limbs, or merged shapes)?
- Distortion, warping, or implausible blending of objects (e.g. melted features, fused geometry) should reduce the score.
- Unusual or surreal objects are acceptable if they are clearly rendered and visually deliberate.

Score from 1 to 5:
5: Clean, realistic, and fully coherent - no visible flaws
4: Mostly clean with minor visual issues or stiffness
3: Noticeable visual flaws (e.g. broken grips, distorted anatomy), but the image is still readable
2: Major visual issues - warped or broken key elements disrupt coherence
1: Severe rendering failure - image appears nonsensical or corrupted

---

Keep each explanation to at most {explanation_max_words} words.

Respond using exactly this format:
### ALIGNMENT SCORE: score
### ALIGNMENT EXPLANATION: explanation
### QUALITY SCORE: score
### QUALITY EXPLANATION: explanation"""


def parse_evaluation_text(text: str) -> dict[str, Any]:
    patterns = {
        "alignment score": r"###\s*ALIGNMENT SCORE:\s*(\d+)",
        "alignment explanation": r"###\s*ALIGNMENT EXPLANATION:\s*(.*?)\s*###\s*QUALITY SCORE:",
        "quality score": r"###\s*QUALITY SCORE:\s*(\d+)",
        "quality explanation": r"###\s*QUALITY EXPLANATION:\s*(.*)",
    }

    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError(f"Failed to parse `{key}` from model response:\n{text}")
        value = match.group(1).strip()
        parsed[key] = _parse_score(value, key, text) if "score" in key else value
    return parsed


def _parse_score(value: str, key: str, full_text: str) -> int:
    score = int(value)
    if 1 <= score <= 5:
        return score
    raise ValueError(f"Parsed `{key}` out of range 1..5: {score}\n{full_text}")


class LocalQwenVLEvaluator:
    def __init__(
        self,
        model_name: str,
        *,
        dtype: str,
        device: str,
        attn_implementation: str,
        min_pixels: int,
        max_pixels: int,
        load_in_4bit: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise SystemExit(
                "Missing local VLM dependencies. Install them in the active environment, for example:\n"
                "pip install torch accelerate 'git+https://github.com/huggingface/transformers' qwen-vl-utils[decord] pillow tqdm\n"
                "Optional for --load-in-4bit: pip install bitsandbytes"
            ) from exc

        torch_dtype = self._torch_dtype(torch, dtype)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device == "auto" else None,
        }
        if attn_implementation and attn_implementation != "none":
            model_kwargs["attn_implementation"] = attn_implementation
        if device != "auto":
            model_kwargs.pop("device_map")

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise SystemExit("Install bitsandbytes to use --load-in-4bit.") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except Exception:
            if attn_implementation == "flash_attention_2":
                model_kwargs.pop("attn_implementation", None)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
            else:
                raise

        if device != "auto":
            self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    @staticmethod
    def _torch_dtype(torch_module: Any, dtype: str) -> Any:
        if dtype == "auto":
            return "auto"
        if dtype == "bfloat16":
            return torch_module.bfloat16
        if dtype == "float16":
            return torch_module.float16
        if dtype == "float32":
            return torch_module.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int,
        explanation_max_words: int,
    ) -> dict[str, Any]:
        return self.evaluate_batch(
            [
                {
                    "image_path": image_path,
                    "prompt": prompt,
                }
            ],
            max_new_tokens=max_new_tokens,
            explanation_max_words=explanation_max_words,
        )[0]

    def evaluate_batch(
        self,
        jobs: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        explanation_max_words: int,
    ) -> list[dict[str, Any]]:
        import torch
        from qwen_vl_utils import process_vision_info

        chat_texts = []
        image_inputs = []
        video_inputs = []

        for job in jobs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(job["image_path"])},
                        {
                            "type": "text",
                            "text": build_eval_prompt(
                                str(job["prompt"]),
                                explanation_max_words=explanation_max_words,
                            ),
                        },
                    ],
                }
            ]
            chat_texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            sample_image_inputs, sample_video_inputs = process_vision_info(messages)
            if sample_image_inputs:
                image_inputs.extend(sample_image_inputs)
            if sample_video_inputs:
                video_inputs.extend(sample_video_inputs)

        inputs = self.processor(
            text=chat_texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        )

        target_device = self._input_device()
        if target_device is not None:
            inputs = inputs.to(target_device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        response_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results = []
        for text in response_text:
            parsed = parse_evaluation_text(text.strip())
            parsed["raw response"] = text.strip()
            results.append(parsed)
        return results

    def _input_device(self) -> str | None:
        if self.device != "auto":
            return self.device
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for module_device in hf_device_map.values():
                if isinstance(module_device, int):
                    return f"cuda:{module_device}"
                if isinstance(module_device, str) and module_device not in {"cpu", "disk", "meta"}:
                    return module_device
        return getattr(self.model, "device", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute local Qwen2.5-VL alignment and quality scores for a whoops50bench-style "
            "experiment directory."
        )
    )
    parser.add_argument(
        "-generations",
        "--generations",
        default="vlm_guidance_project/whoops50bench",
        help="Root directory created by run_bench.sh.",
    )
    parser.add_argument(
        "--output",
        default="metrics/alignment_local_whoops50bench.csv",
        help="Detailed per-image CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="metrics/alignment_local_whoops50bench_summary.csv",
        help="Aggregated per-pipeline CSV output path.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--explanation-max-words",
        type=int,
        default=18,
        help="Short explanation budget per metric. Lower values make generation faster.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of image/prompt pairs per model.generate call.",
    )
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="Use auto, cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention backend passed to transformers. Use sdpa by default; use none to let transformers choose.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes NF4 4-bit loading.")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit on number of images.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write partial CSV after this many batches. Use larger values for slightly less I/O overhead.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse rows already present in --output.")
    parser.add_argument(
        "--no-image-dedupe",
        action="store_true",
        help="Disable reusing scores for identical image bytes with the same prompt.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write failed rows with empty scores instead of stopping.",
    )
    parser.add_argument(
        "--skip-pipelines",
        nargs="*",
        default=(),
        help="Pipeline directory names to skip, for example vanilla_sd.",
    )
    parser.add_argument(
        "--only-pipelines",
        nargs="*",
        default=None,
        help="Only evaluate these pipeline directory names.",
    )
    return parser.parse_args()


def build_jobs(root: Path, *, skip_pipelines: set[str], only_pipelines: set[str] | None) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Generations root not found: {root}")

    jobs: list[dict[str, Any]] = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for prompt_fallback_index, prompt_dir in enumerate(sorted(path for path in experiment_dir.iterdir() if path.is_dir())):
            prompt_index = parse_prompt_index(prompt_dir.name, prompt_fallback_index)
            for pipeline_dir in sorted(path for path in prompt_dir.iterdir() if path.is_dir()):
                pipeline = pipeline_dir.name
                if pipeline in skip_pipelines:
                    continue
                if only_pipelines is not None and pipeline not in only_pipelines:
                    continue
                prompt_path = pipeline_dir / "prompt.txt"
                if not prompt_path.is_file():
                    prompt_path = prompt_dir / "prompt.txt"
                try:
                    prompt = read_prompt(prompt_path, prompt_dir)
                    image_path = find_image_file(pipeline_dir)
                except FileNotFoundError as exc:
                    print(f"Skipping incomplete pipeline directory: {exc}", file=sys.stderr)
                    continue
                image_hash = sha256_file(image_path)
                jobs.append(
                    {
                        "experiment": experiment_dir.name,
                        "pipeline": pipeline,
                        "pipeline_key": f"{experiment_dir.name}/{pipeline}",
                        "prompt_index": prompt_index,
                        "prompt_dir": prompt_dir.name,
                        "prompt": prompt,
                        "image_hash": image_hash,
                        "image_path": str(image_path),
                        "prompt_path": str(prompt_path) if prompt_path.is_file() else "",
                    }
                )
    return jobs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prompt_index(prompt_dir_name: str, fallback_index: int) -> int:
    prefix = prompt_dir_name.split("_", maxsplit=1)[0]
    return int(prefix) if prefix.isdigit() else fallback_index


def read_prompt(prompt_path: Path, prompt_dir: Path) -> str:
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip()

    summary_path = prompt_dir / "run_summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for value in data.values():
            if not isinstance(value, dict):
                continue
            run = value.get("run")
            if isinstance(run, dict) and run.get("prompt"):
                return str(run["prompt"]).strip()
            if value.get("prompt"):
                return str(value["prompt"]).strip()

    raise FileNotFoundError(f"No prompt.txt or prompt in run_summary.json for {prompt_dir}")


def find_image_file(pipeline_dir: Path) -> Path:
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
        raise ValueError(f"Multiple image files found in {pipeline_dir}: {[path.name for path in image_files]}")
    return image_files[0]


def load_completed_rows(output_path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not output_path.is_file():
        return {}

    completed = {}
    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("alignment_score") and row.get("quality_score"):
                if not row.get("image_hash") and row.get("image_path"):
                    image_path = Path(row["image_path"])
                    if image_path.is_file():
                        row["image_hash"] = sha256_file(image_path)
                completed[job_key(row)] = row
    return completed


def job_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["experiment"]),
        str(row["pipeline"]),
        str(row["prompt_index"]),
        str(row["image_path"]),
    )


def evaluate_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_path = Path(args.output)
    completed = load_completed_rows(output_path) if args.resume else {}
    rows = list(completed.values())
    pending = [job for job in jobs if job_key(job) not in completed]

    if args.limit is not None:
        pending = pending[: args.limit]

    duplicate_jobs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not args.no_image_dedupe:
        rows, pending, duplicate_jobs = dedupe_pending_jobs(rows, pending, args.model)

    if not pending:
        return normalize_rows(rows)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.save_every < 1:
        raise SystemExit("--save-every must be at least 1.")

    evaluator = LocalQwenVLEvaluator(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
    )

    batches = list(chunked(pending, args.batch_size))
    for batch_index, batch in enumerate(tqdm(batches, desc="Local alignment batches"), start=1):
        batch_rows = evaluate_batch_with_fallback(evaluator, batch, args)
        rows.extend(batch_rows)
        for row in batch_rows:
            for duplicate_job in duplicate_jobs.get(dedupe_key(row), []):
                rows.append(copy_scores_to_job(row, duplicate_job))
        if batch_index % args.save_every == 0:
            write_detail_csv(output_path, normalize_rows(rows))

    return normalize_rows(rows)


def dedupe_pending_jobs(
    rows: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    score_cache = build_score_cache(rows)
    representative_jobs: list[dict[str, Any]] = []
    representative_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_jobs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    reused_count = 0
    current_duplicate_count = 0

    for job in pending:
        key = dedupe_key(job)
        cached_row = score_cache.get(key)
        if cached_row is not None:
            rows.append(copy_scores_to_job(cached_row, job, model=model))
            reused_count += 1
            continue
        if key in representative_by_key:
            duplicate_jobs[key].append(job)
            current_duplicate_count += 1
            continue
        representative_by_key[key] = job
        representative_jobs.append(job)

    if reused_count or current_duplicate_count:
        print(
            "Image dedupe: "
            f"reused {reused_count} rows from existing CSV, "
            f"will copy {current_duplicate_count} rows from this run, "
            f"will evaluate {len(representative_jobs)} unique image/prompt pairs.",
            file=sys.stderr,
        )

    return rows, representative_jobs, duplicate_jobs


def build_score_cache(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    cache = {}
    for row in rows:
        if not row.get("alignment_score") or not row.get("quality_score"):
            continue
        key = dedupe_key(row)
        if key[0] and key[1]:
            cache[key] = row
    return cache


def dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("image_hash") or ""), str(row.get("prompt") or "")


def copy_scores_to_job(source_row: dict[str, Any], job: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    return {
        **job,
        "alignment_score": source_row["alignment_score"],
        "alignment_explanation": source_row["alignment_explanation"],
        "quality_score": source_row["quality_score"],
        "quality_explanation": source_row["quality_explanation"],
        "model": model or source_row.get("model", ""),
        "raw_response": source_row.get("raw_response", ""),
    }


def evaluate_batch_with_fallback(
    evaluator: LocalQwenVLEvaluator,
    batch: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        results = evaluator.evaluate_batch(
            batch,
            max_new_tokens=args.max_new_tokens,
            explanation_max_words=args.explanation_max_words,
        )
        return [make_success_row(job, result, args.model) for job, result in zip(batch, results)]
    except Exception as exc:
        if len(batch) > 1:
            midpoint = len(batch) // 2
            return [
                *evaluate_batch_with_fallback(evaluator, batch[:midpoint], args),
                *evaluate_batch_with_fallback(evaluator, batch[midpoint:], args),
            ]
        if not args.continue_on_error:
            raise
        return [make_error_row(batch[0], exc, args.model)]


def make_success_row(job: dict[str, Any], result: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        **job,
        "alignment_score": result["alignment score"],
        "alignment_explanation": result["alignment explanation"],
        "quality_score": result["quality score"],
        "quality_explanation": result["quality explanation"],
        "model": model,
        "raw_response": result["raw response"],
    }


def make_error_row(job: dict[str, Any], exc: Exception, model: str) -> dict[str, Any]:
    return {
        **job,
        "alignment_score": "",
        "alignment_explanation": f"ERROR: {exc}",
        "quality_score": "",
        "quality_explanation": f"ERROR: {exc}",
        "model": model,
        "raw_response": "",
    }


def chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append({field: row.get(field, "") for field in DETAIL_FIELDNAMES})
    normalized.sort(key=lambda row: (str(row["pipeline_key"]), int(row["prompt_index"]), str(row["image_path"])))
    return normalized


def write_detail_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETAIL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["alignment_score"] == "" or row["quality_score"] == "":
            continue
        grouped[str(row["pipeline_key"])].append(row)

    summary = []
    for pipeline_key, group in sorted(grouped.items()):
        alignment = [float(row["alignment_score"]) for row in group]
        quality = [float(row["quality_score"]) for row in group]
        first = group[0]
        summary.append(
            {
                "pipeline_key": pipeline_key,
                "experiment": first["experiment"],
                "pipeline": first["pipeline"],
                "n": len(group),
                "alignment_mean": mean(alignment),
                "alignment_std": std(alignment),
                "quality_mean": mean(quality),
                "quality_std": std(quality),
            }
        )
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def write_summary_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(
        Path(args.generations),
        skip_pipelines=set(args.skip_pipelines),
        only_pipelines=set(args.only_pipelines) if args.only_pipelines is not None else None,
    )
    if not jobs:
        raise SystemExit(f"No image jobs found under {args.generations}")

    print(f"Found {len(jobs)} image jobs under {args.generations}", file=sys.stderr)
    print_job_overview(jobs)
    rows = evaluate_jobs(args, jobs)
    write_detail_csv(Path(args.output), rows)

    summary_rows = summarize_rows(rows)
    write_summary_csv(Path(args.summary_output), summary_rows)

    print(f"Saved {len(rows)} detailed rows to {args.output}")
    print(f"Saved {len(summary_rows)} pipeline summary rows to {args.summary_output}")


def print_job_overview(jobs: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for job in jobs:
        counts[str(job["pipeline_key"])] += 1

    print(f"Found {len(counts)} experiment/pipeline groups:", file=sys.stderr)
    for pipeline_key, count in sorted(counts.items()):
        print(f"  {pipeline_key}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
