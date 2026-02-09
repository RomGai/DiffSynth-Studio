"""Qwen3-VL to diffusion conditioning pipeline.

This file extracts the multimodal LLM -> diffusion conditioning flow into a
single, self-contained pipeline module. It focuses on:
1) Producing token embeddings + mask from reference image + instruction.
2) Packaging diffusion input dicts (what DiT/UNet expects).
3) Defining connector/adapter modules that bridge LLM embeddings to DiT.

Note: This does NOT implement the DiT injection itself. It only outputs the
connector-adapted tensors that a DiT would consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


to_pil = ToPILImage()


@dataclass
class Qwen3VLPipelineConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Thinking"
    max_length: int = 640
    dtype: torch.dtype | str = "auto"
    device_map: str = "auto"
    attn_implementation: Optional[str] = None


def build_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You will see the user's shopping history (product images + descriptions). "
                "Predict the most likely next product types the user will purchase, with "
                "detailed descriptions. Do not list the past purchases. Output in English."
            ),
        }
    ]
    for item in history:
        content.append({"type": "image", "image": item["image"]})
        content.append({"type": "text", "text": item["description"]})
    return [{"role": "user", "content": content}]


class Qwen3VLTextImageEmbedder(nn.Module):
    """Encode (reference image + instruction) into token embeddings + mask."""

    def __init__(self, config: Qwen3VLPipelineConfig):
        super().__init__()
        model_kwargs: Dict[str, Any] = {
            "dtype": config.dtype,
            "device_map": config.device_map,
        }
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(config.model_name_or_path)
        self.max_length = config.max_length

    def forward(
        self,
        history: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        formatted_history: List[Dict[str, Any]] = []
        for item in history:
            image = item["image"]
            if torch.is_tensor(image):
                image = to_pil(image)
            formatted_history.append(
                {
                    "image": image,
                    "description": item["description"],
                }
            )

        messages = build_messages(formatted_history)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        attn_mask = inputs.attention_mask

        hidden = hidden[:, : self.max_length, :]
        attn_mask = attn_mask[:, : self.max_length]
        return hidden, attn_mask


class TokenRefiner(nn.Module):
    """Lightweight token refiner for LLM embeddings."""

    def __init__(self, in_channels: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask == 0
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(x)
        return x


class Qwen3VLConnector(nn.Module):
    """Connector that maps Qwen3-VL embeddings to DiT-friendly tensors."""

    def __init__(self, in_channels: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.refiner = TokenRefiner(in_channels, hidden_size, num_heads)
        self.global_proj = nn.Linear(in_channels, hidden_size)

    def forward(
        self, llm_embeddings: torch.Tensor, masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        mask_float = masks.unsqueeze(-1).to(llm_embeddings.dtype)
        pooled = (llm_embeddings * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        refined_tokens = self.refiner(llm_embeddings, masks)
        global_vector = self.global_proj(pooled)
        return {
            "encoder_hidden_states": refined_tokens,
            "global_vector": global_vector,
        }


class DiffusionInputBuilder:
    """Build diffusion-ready input dictionaries (no DiT implementation here)."""

    def __init__(self, connector: Qwen3VLConnector):
        self.connector = connector

    def build_inputs(
        self,
        llm_embeddings: torch.Tensor,
        llm_masks: torch.Tensor,
        txt_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        target_device = llm_embeddings.device
        self.connector = self.connector.to(device=target_device, dtype=llm_embeddings.dtype)
        connector_out = self.connector(llm_embeddings, llm_masks)
        diffusion_inputs = {
            "llm_embedding": llm_embeddings,
            "mask": llm_masks,
            "encoder_hidden_states": connector_out["encoder_hidden_states"],
            "global_vector": connector_out["global_vector"],
        }
        if txt_ids is not None:
            diffusion_inputs["txt_ids"] = txt_ids
        return diffusion_inputs


class Qwen3VLDiffusionPipeline:
    """End-to-end pipeline: Qwen3-VL -> connector -> diffusion input dict."""

    def __init__(
        self,
        config: Qwen3VLPipelineConfig,
        connector_in_channels: int = 4096,
        connector_hidden_size: int = 3072,
        connector_num_heads: int = 24,
    ) -> None:
        self.embedder = Qwen3VLTextImageEmbedder(config)
        self.connector = Qwen3VLConnector(
            in_channels=connector_in_channels,
            hidden_size=connector_hidden_size,
            num_heads=connector_num_heads,
        )
        self.input_builder = DiffusionInputBuilder(self.connector)

    def encode(
        self, history: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embedder(history)

    def build_diffusion_inputs(
        self,
        history: List[Dict[str, Any]],
        txt_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings, masks = self.encode(history)
        return self.input_builder.build_inputs(
            llm_embeddings=embeddings,
            llm_masks=masks,
            txt_ids=txt_ids,
        )

    @staticmethod
    def load_sample_history(json_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        history: List[Dict[str, Any]] = []
        for record in records:
            prompt = record.get("description")
            image_path = record.get("image")
            if prompt is None or image_path is None:
                raise ValueError("Each record must contain 'image' and 'description'.")
            image_file = Path(image_path)
            if not image_file.is_absolute():
                image_file = path.parent / image_file
            history.append(
                {
                    "image": Image.open(image_file).convert("RGB"),
                    "description": prompt,
                }
            )
        return history

    def build_diffusion_inputs_from_history(
        self, json_path: str | Path, txt_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        history = self.load_sample_history(json_path)
        return self.build_diffusion_inputs(history, txt_ids=txt_ids)


def main() -> None:
    json_path = Path("sample_history.json")
    config = Qwen3VLPipelineConfig()
    pipeline = Qwen3VLDiffusionPipeline(config)
    diffusion_inputs = pipeline.build_diffusion_inputs_from_history(json_path)
    print("diffusion_inputs keys:", list(diffusion_inputs.keys()))
    for key, value in diffusion_inputs.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
