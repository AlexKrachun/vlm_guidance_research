from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from vlm_guidance.utils.io import save_json


@dataclass
class GenerationResult:
    pipeline_name: str
    image_paths: List[str]
    metadata: Dict[str, Any]


class BaseText2ImagePipeline(ABC):
    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        raise NotImplementedError


class Text2ImageRunner:
    def __init__(self, pipeline: BaseText2ImagePipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        run_dir: str | Path,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
        image_filename_template: str = "img_{index:02d}.png",
        prompt_filename: str = "prompt.txt",
        summary_filename: str = "result_summary.json",
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / prompt_filename).write_text(prompt, encoding="utf-8")

        images = self.pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_size=batch_size,
        )

        image_paths: List[str] = []
        for idx, image in enumerate(images):
            filename = image_filename_template.format(index=idx)
            path = run_dir / filename
            image.save(path)
            image_paths.append(str(path))

        result = {
            "pipeline_name": self.pipeline.pipeline_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "batch_size": batch_size,
            "image_paths": image_paths,
        }
        save_json(result, run_dir / summary_filename)
        return result
