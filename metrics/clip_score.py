import argparse
import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import open_clip
import torch
from PIL import Image


ImageInput = Union[str, Path, Image.Image, torch.Tensor, np.ndarray]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, Path):
        image = str(image)

    if isinstance(image, str):
        if image.endswith(".npy"):
            array = np.load(image)
            return Image.fromarray(array[:, :, [2, 1, 0]], "RGB")
        return Image.open(image).convert("RGB")

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Expected a single image tensor, got a batch.")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("Expected image tensor with shape [C, H, W] or [1, C, H, W].")
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.clamp(0, 1).mul(255).byte().numpy()
        if tensor.shape[-1] == 1:
            tensor = np.repeat(tensor, 3, axis=-1)
        return Image.fromarray(tensor, "RGB")

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim != 3:
            raise ValueError("Expected image array with shape [H, W, C].")
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array, "RGB")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


class CLIPScorePipeline:
    """
    Standalone CLIP score pipeline based on t2v_metrics/models/clipscore_models/clip_model.py.
    Computes cosine similarity between normalized image and text embeddings.
    """

    def __init__(
        self,
        model_name: str = "openai:ViT-L-14",
        device: str | None = None,
        cache_dir: str = "./hf_cache/",
    ) -> None:
        self.model_name = model_name
        self.device = device or _default_device()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        pretrained, arch = model_name.split(":", maxsplit=1)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch,
            pretrained=pretrained,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model.eval()

    def _encode_images(self, images: Sequence[ImageInput]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(_load_image(image)) for image in images], dim=0)
        batch = batch.to(self.device)
        features = self.model.encode_image(batch)
        return features / features.norm(dim=-1, keepdim=True)

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(self, image: ImageInput, prompt: str) -> float:
        image_features = self._encode_images([image])
        text_features = self._encode_texts([prompt])
        return float((image_features * text_features).sum(dim=-1).item())

    @torch.no_grad()
    def score_pairs(self, images: Sequence[ImageInput], prompts: Sequence[str]) -> torch.Tensor:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")
        image_features = self._encode_images(images)
        text_features = self._encode_texts(prompts)
        return (image_features * text_features).sum(dim=-1)

    @torch.no_grad()
    def score_matrix(self, images: Sequence[ImageInput], prompts: Sequence[str]) -> torch.Tensor:
        image_features = self._encode_images(images)
        text_features = self._encode_texts(prompts)
        return image_features @ text_features.T


def compute_clip_score(
    image: ImageInput,
    prompt: str,
    model_name: str = "openai:ViT-L-14",
    device: str | None = None,
    cache_dir: str = "./hf_cache/",
) -> float:
    pipeline = CLIPScorePipeline(model_name=model_name, device=device, cache_dir=cache_dir)
    return pipeline.score(image, prompt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLIP score between an image and a text prompt.")
    parser.add_argument("--image", required=True, help="Path to an image file or .npy array.")
    parser.add_argument("--prompt", required=True, help="Text prompt to compare with the image.")
    parser.add_argument("--model", default="openai:ViT-L-14", help="open_clip model id in PRETRAINED:ARCH format.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--cache-dir", default="./hf_cache/", help="Directory for downloaded model weights.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = CLIPScorePipeline(
        model_name=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    score = pipeline.score(args.image, args.prompt)
    print(f"CLIP score: {score:.6f}")


if __name__ == "__main__":
    main()
    # прогнать на картинках сгенеренных по simple_cases.txt и comples_cases.txt
    
"""
python clip_score.py --image path/to/image.png --prompt "a book on the sofa"
"""