from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from t2v_metrics.models.vqascore_models import clip_t5_model

from vlm_guidance_project.vlm_guidance.scorers.base import BaseDifferentiableScorer, ScoreOutput
from vlm_guidance_project.vlm_guidance.utils.amp import PrecisionConfig, autocast_context
from vlm_guidance_project.vlm_guidance.utils.images import to_bchw_float01


_QWEN25_VL_3B_ALIASES = {
    "qwen-2-5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen/qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
}


@dataclass
class CLIPFlanT5ScorerConfig:
    model_name: str = "clip-flant5-xl"
    device: str = "cuda:0"
    autocast_enabled: bool = False
    autocast_dtype: str = "bfloat16"
    question_template: str = 'Does this figure show "{}"? Please answer yes or no.'


class CLIPFlanT5DifferentiableScorer(BaseDifferentiableScorer):
    def __init__(
        self,
        model_name: str = "clip-flant5-xl",
        device: str = "cuda:0",
        autocast_enabled: bool = False,
        autocast_dtype: str = "bfloat16",
        question_template: str = 'Does this figure show "{}"? Please answer yes or no.',
    ) -> None:
        super().__init__(device=device if torch.cuda.is_available() else "cpu")
        self.precision = PrecisionConfig(enabled=autocast_enabled and self.device.type == "cuda", dtype=autocast_dtype)
        self.default_question_template = question_template
        self.model_family = "qwen25_vl" if model_name.lower() in _QWEN25_VL_3B_ALIASES else "clip_flant5"

        if self.model_family == "clip_flant5":
            self._init_clip_flant5(model_name=model_name)
        else:
            self._init_qwen25_vl(model_name=model_name)

    def _init_clip_flant5(self, model_name: str) -> None:
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
        self._yes_token_id, self._no_token_id, self._answer_token_position = self._prepare_binary_answer_tokens()

    def _init_qwen25_vl(self, model_name: str) -> None:
        hf_model_name = _QWEN25_VL_3B_ALIASES[model_name.lower()]
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(hf_model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_name,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer("image_mean", torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.patch_size = int(self.image_processor.patch_size)
        self.merge_size = int(self.image_processor.merge_size)
        self.temporal_patch_size = int(self.image_processor.temporal_patch_size)
        self.min_pixels = int(self.image_processor.min_pixels)
        self.max_pixels = int(self.image_processor.max_pixels)

        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError("Qwen2.5-VL scorer expects single-token 'Yes'/'No' answers.")
        self._yes_token_id = int(yes_ids[0])
        self._no_token_id = int(no_ids[0])

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
        if self.model_family != "clip_flant5":
            raise RuntimeError("preprocess_image is only used by the CLIP-FlanT5 scorer path.")
        x = to_bchw_float01(image).to(self.device)
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        x = (x - self.image_mean.to(x.device, x.dtype)) / self.image_std.to(x.device, x.dtype)
        return x

    def _preprocess_qwen_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = to_bchw_float01(image).to(self.device)
        x = x * 255.0

        flat_patches: list[torch.Tensor] = []
        grid_thw: list[list[int]] = []
        for i in range(x.shape[0]):
            img = x[i : i + 1]
            _, _, height, width = img.shape
            resized_height, resized_width = smart_resize(
                int(height),
                int(width),
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
            img = (img / 255.0 - self.image_mean.to(img.device, img.dtype)) / self.image_std.to(img.device, img.dtype)

            if img.shape[0] % self.temporal_patch_size != 0:
                repeats = img[-1:].repeat(self.temporal_patch_size - img.shape[0] % self.temporal_patch_size, 1, 1, 1)
                img = torch.cat([img, repeats], dim=0)

            grid_t = img.shape[0] // self.temporal_patch_size
            grid_h = resized_height // self.patch_size
            grid_w = resized_width // self.patch_size
            channel = img.shape[1]

            patches = img.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
            flat = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
            flat_patches.append(flat)
            grid_thw.append([grid_t, grid_h, grid_w])

        pixel_values = torch.cat(flat_patches, dim=0)
        image_grid_thw = torch.tensor(grid_thw, device=self.device, dtype=torch.long)
        return pixel_values, image_grid_thw

    def _build_qwen_text_inputs(
        self,
        prompt: Union[str, Sequence[str]],
        question_template: str,
        batch_size: int,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        if len(texts) != batch_size:
            raise ValueError("Number of prompts must match batch size.")
        prompts = []
        for text in texts:
            question = question_template.format(text)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            prompts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        merge_length = self.image_processor.merge_size**2
        image_token = self.processor.image_token
        expanded_prompts: list[str] = []
        for i, text in enumerate(prompts):
            repeat_count = int(image_grid_thw[i].prod().item()) // merge_length
            expanded = text.replace(image_token, "<|placeholder|>" * repeat_count, 1)
            expanded = expanded.replace("<|placeholder|>", image_token)
            expanded_prompts.append(expanded)

        tokenized = self.tokenizer(expanded_prompts, padding=True, return_tensors="pt")
        return tokenized.input_ids.to(self.device), tokenized.attention_mask.to(self.device)

    def _tokenize_answers(self, answers: Sequence[str]) -> torch.Tensor:
        labels = [self.t5_tokenizer_image_token(answer, self.tokenizer, return_tensors="pt") for answer in answers]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)
        return labels[:, : self.tokenizer.model_max_length].to(self.device)

    def _prepare_binary_answer_tokens(self) -> Tuple[int, int, int]:
        yes_answer = self.format_answer("Yes", conversation_style=self.conversational_style)
        no_answer = self.format_answer("No", conversation_style=self.conversational_style)
        yes_labels = self._tokenize_answers([yes_answer])[0]
        no_labels = self._tokenize_answers([no_answer])[0]

        yes_valid = yes_labels.ne(self.IGNORE_INDEX)
        no_valid = no_labels.ne(self.IGNORE_INDEX)
        if yes_valid.sum().item() != no_valid.sum().item():
            raise ValueError("Formatted yes/no answers must have the same tokenized length.")

        differing_positions = torch.nonzero((yes_labels != no_labels) & yes_valid & no_valid, as_tuple=False).flatten()
        if differing_positions.numel() != 1:
            raise ValueError("Formatted yes/no answers must differ by exactly one token.")

        answer_token_position = int(differing_positions.item())
        yes_token_id = int(yes_labels[answer_token_position].item())
        no_token_id = int(no_labels[answer_token_position].item())
        return yes_token_id, no_token_id, answer_token_position

    def _build_inputs_and_labels(
        self,
        batch_size: int,
        prompt: Union[str, Sequence[str]],
        question_template: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        if len(texts) != batch_size:
            raise ValueError("Number of prompts must match batch size.")

        questions = [self.format_question(question_template.format(text), conversation_style=self.conversational_style) for text in texts]
        answers = [self.format_answer("Yes", conversation_style=self.conversational_style) for _ in texts]

        input_ids = [self.t5_tokenizer_image_token(q, self.tokenizer, return_tensors="pt") for q in questions]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        input_ids = input_ids[:, : self.tokenizer.model_max_length].to(self.device)
        labels = self._tokenize_answers(answers)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        decoder_attention_mask = labels.ne(self.IGNORE_INDEX).to(self.device)
        return input_ids, attention_mask, decoder_attention_mask, labels

    def generate_answer(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, Sequence[str]],
        question_template: str | None = None,
        max_new_tokens: int = 16,
    ) -> list[str]:
        question_template = question_template or self.default_question_template

        if self.model_family == "clip_flant5":
            images = self.preprocess_image(image)
            texts = [prompt] * images.shape[0] if isinstance(prompt, str) else list(prompt)
            if len(texts) != images.shape[0]:
                raise ValueError("Number of prompts must match batch size.")

            questions = [self.format_question(question_template.format(text), conversation_style=self.conversational_style) for text in texts]
            input_ids = [self.t5_tokenizer_image_token(q, self.tokenizer, return_tensors="pt") for q in questions]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            input_ids = input_ids[:, : self.tokenizer.model_max_length].to(self.device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)

            decoder_start_token_id = self.model.config.decoder_start_token_id
            if decoder_start_token_id is None:
                raise ValueError("decoder_start_token_id must be set for CLIP-FlanT5 generation.")
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else decoder_start_token_id

            _, attention_mask, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=None,
                past_key_values=None,
                labels=None,
                images=images,
            )

            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                fill_value=decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            max_new_tokens = max(1, int(max_new_tokens))

            with torch.no_grad():
                with autocast_context(self.device, self.precision):
                    for _ in range(max_new_tokens):
                        outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            use_cache=False,
                            return_dict=True,
                        )
                        next_token_ids = outputs.logits[:, -1, :].float().argmax(dim=-1)
                        if eos_token_id is not None:
                            next_token_ids = torch.where(finished, torch.full_like(next_token_ids, pad_token_id), next_token_ids)
                            finished = finished | next_token_ids.eq(eos_token_id)

                        decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                        if finished.all():
                            break

            decoded = self.tokenizer.batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)
            return [text.strip() for text in decoded]

        pixel_values, image_grid_thw = self._preprocess_qwen_image(image)
        input_ids, attention_mask = self._build_qwen_text_inputs(
            prompt=prompt,
            question_template=question_template,
            batch_size=image_grid_thw.shape[0],
            image_grid_thw=image_grid_thw,
        )

        with torch.no_grad():
            with autocast_context(self.device, self.precision):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values.to(dtype=next(self.model.parameters()).dtype),
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=max(1, int(max_new_tokens)),
                    use_cache=False,
                )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        decoded = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [text.strip() for text in decoded]

    def forward(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, Sequence[str]],
        question_template: str | None = None,
        yes_no_loss: bool = True,
        **_: Any,
    ) -> ScoreOutput:
        question_template = question_template or self.default_question_template

        if self.model_family == "clip_flant5":
            images = self.preprocess_image(image)
            input_ids, attention_mask, decoder_attention_mask, labels = self._build_inputs_and_labels(
                batch_size=images.shape[0],
                prompt=prompt,
                question_template=question_template,
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

            token_logits = logits[:, self._answer_token_position, :].float()
        else:
            pixel_values, image_grid_thw = self._preprocess_qwen_image(image)
            input_ids, attention_mask = self._build_qwen_text_inputs(
                prompt=prompt,
                question_template=question_template,
                batch_size=image_grid_thw.shape[0],
                image_grid_thw=image_grid_thw,
            )
            with autocast_context(self.device, self.precision):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values.to(dtype=next(self.model.parameters()).dtype),
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits

            token_logits = logits[:, -1, :].float()

        token_probs = token_logits.softmax(dim=-1)
        p_yes = token_probs[:, self._yes_token_id]
        p_no = token_probs[:, self._no_token_id]
        prob_margin = p_yes - p_no

        score = torch.sigmoid(prob_margin)
        loss_type = "yes_no_margin" if yes_no_loss else "yes_cross_entropy"
        if yes_no_loss:
            loss = -F.logsigmoid(prob_margin)
        else:
            target = torch.full(
                (token_logits.shape[0],),
                fill_value=self._yes_token_id,
                dtype=torch.long,
                device=token_logits.device,
            )
            loss = F.cross_entropy(token_logits, target, reduction="none")
        return ScoreOutput(
            score=score,
            loss=loss,
            extras={
                "loss_type": loss_type,
                "p_yes": p_yes,
                "p_no": p_no,
                "prob_margin": prob_margin,
                "answer_token_position": torch.full_like(
                    p_yes,
                    self._answer_token_position if self.model_family == "clip_flant5" else input_ids.shape[1],
                    dtype=torch.long,
                ),
            },
        )
