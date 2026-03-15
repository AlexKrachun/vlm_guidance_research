from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from t2v_metrics.models.vqascore_models import clip_t5_model

from vlm_guidance.scorers.base import BaseDifferentiableScorer, ScoreOutput
from vlm_guidance.utils.amp import PrecisionConfig, autocast_context
from vlm_guidance.utils.images import to_bchw_float01


@dataclass
class CLIPFlanT5ScorerConfig:
    model_name: str = "clip-flant5-xl"
    device: str = "cuda:0"
    autocast_enabled: bool = False
    autocast_dtype: str = "bfloat16"
    question_template: str = 'Does this figure show "{}"? Please answer yes or no.'
    answer_template: str = "Yes"


class CLIPFlanT5DifferentiableScorer(BaseDifferentiableScorer):
    def __init__(
        self,
        model_name: str = "clip-flant5-xl",
        device: str = "cuda:0",
        autocast_enabled: bool = False,
        autocast_dtype: str = "bfloat16",
        question_template: str = 'Does this figure show "{}"? Please answer yes or no.',
        answer_template: str = "Yes",
    ) -> None:
        super().__init__(device=device if torch.cuda.is_available() else "cpu")
        self.precision = PrecisionConfig(enabled=autocast_enabled and self.device.type == "cuda", dtype=autocast_dtype)
        self.default_question_template = question_template
        self.default_answer_template = answer_template

        self.CLIPT5Model = clip_t5_model.CLIPT5Model
        self.format_question = clip_t5_model.format_question
        self.format_answer = clip_t5_model.format_answer
        self.t5_tokenizer_image_token = clip_t5_model.t5_tokenizer_image_token
        self.IGNORE_INDEX = clip_t5_model.IGNORE_INDEX

        self.backend = self.CLIPT5Model(model_name=model_name, device=str(self.device))
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.image_processor = self.backend.image_processor
        self.conversational_style = self.backend.conversational_style

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._patch_vision_tower_forward()

        self.register_buffer("image_mean", torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        crop_size = getattr(self.image_processor, "crop_size", None)
        size = getattr(self.image_processor, "size", None)
        if isinstance(crop_size, dict):
            target_h = crop_size.get("height", crop_size.get("shortest_edge", 336))
            target_w = crop_size.get("width", crop_size.get("shortest_edge", 336))
        elif isinstance(size, dict):
            target_h = size.get("height", size.get("shortest_edge", 336))
            target_w = size.get("width", size.get("shortest_edge", 336))
        elif isinstance(size, int):
            target_h = target_w = size
        else:
            target_h = target_w = 336
        self.target_size = (int(target_h), int(target_w))

    def _patch_vision_tower_forward(self) -> None:
        vision_tower = self.model.get_vision_tower()

        def differentiable_forward(this, images):
            tower_dtype = this.dtype
            image_forward_outs = this.vision_tower(
                images.to(device=this.device, dtype=tower_dtype),
                output_hidden_states=True,
            )
            image_features = this.feature_select(image_forward_outs).to(dtype=tower_dtype)
            return image_features

        vision_tower.forward = types.MethodType(differentiable_forward, vision_tower)

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = to_bchw_float01(image).to(self.device)
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        x = (x - self.image_mean.to(x.device, x.dtype)) / self.image_std.to(x.device, x.dtype)
        return x

    def _build_inputs_and_labels(
        self,
        batch_size: int,
        prompt: Union[str, Sequence[str]],
        question_template: str,
        answer_template: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        if len(texts) != batch_size:
            raise ValueError("Number of prompts must match batch size.")

        questions = [self.format_question(question_template.format(text), conversation_style=self.conversational_style) for text in texts]
        answers = [self.format_answer(answer_template.format(text), conversation_style=self.conversational_style) for text in texts]

        input_ids = [self.t5_tokenizer_image_token(q, self.tokenizer, return_tensors="pt") for q in questions]
        labels = [self.t5_tokenizer_image_token(a, self.tokenizer, return_tensors="pt") for a in answers]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)

        input_ids = input_ids[:, : self.tokenizer.model_max_length].to(self.device)
        labels = labels[:, : self.tokenizer.model_max_length].to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        decoder_attention_mask = labels.ne(self.IGNORE_INDEX).to(self.device)
        return input_ids, attention_mask, decoder_attention_mask, labels

    def forward(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, Sequence[str]],
        question_template: str | None = None,
        answer_template: str | None = None,
        **_: Any,
    ) -> ScoreOutput:
        question_template = question_template or self.default_question_template
        answer_template = answer_template or self.default_answer_template

        images = self.preprocess_image(image)
        input_ids, attention_mask, decoder_attention_mask, labels = self._build_inputs_and_labels(
            batch_size=images.shape[0],
            prompt=prompt,
            question_template=question_template,
            answer_template=answer_template,
        )

        with autocast_context(self.device, self.precision):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                images=images,
                return_dict=True,
            )
            logits = outputs.logits

        vocab = logits.shape[-1]
        per_token_ce = F.cross_entropy(
            logits.float().reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=self.IGNORE_INDEX,
            reduction="none",
        ).view(labels.shape[0], labels.shape[1])

        valid = labels.ne(self.IGNORE_INDEX)
        ce = (per_token_ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        score = torch.exp(-ce)
        return ScoreOutput(score=score, loss=ce, extras={"valid_tokens": valid.sum(dim=1)})
