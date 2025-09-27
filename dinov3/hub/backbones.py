"""Minimal DINOv3 backbone loaders for the few-shot segmentation project."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import torch

from .utils import DINOV3_BASE_URL


class Weights(Enum):
    """Available pretraining sources for the locally vendored models."""

    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"https", "file"}


def convert_path_or_url_to_url(path: str) -> str:
    """Return a `file://` or remote URL that `torch.hub.load_state_dict_from_url` accepts."""

    if _is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_url(
    *,
    compact_arch_name: str,
    patch_size: int,
    weights: Union[Weights, str],
    hash_suffix: Optional[str],
) -> str:
    model_name = "dinov3"
    model_arch = f"{compact_arch_name}{patch_size}"
    weights_name = weights.value.lower() if isinstance(weights, Weights) else Path(weights).name
    hash_part = f"-{hash_suffix}" if hash_suffix else ""
    filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{hash_part}.pth"
    return os.path.join(DINOV3_BASE_URL, f"{model_name}_{model_arch}", filename)


def _load_vit(
    *,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    ffn_ratio: float,
    norm_layer: str,
    **kwargs,
):
    from ..models.vision_transformer import DinoVisionTransformer

    model = DinoVisionTransformer(
        img_size=kwargs.pop("img_size", 224),
        patch_size=patch_size,
        in_chans=kwargs.pop("in_chans", 3),
        pos_embed_rope_base=kwargs.pop("pos_embed_rope_base", 100.0),
        pos_embed_rope_normalize_coords=kwargs.pop("pos_embed_rope_normalize_coords", "separate"),
        pos_embed_rope_rescale_coords=kwargs.pop("pos_embed_rope_rescale_coords", 2.0),
        pos_embed_rope_dtype=kwargs.pop("pos_embed_rope_dtype", "fp32"),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=kwargs.pop("qkv_bias", True),
        drop_path_rate=kwargs.pop("drop_path_rate", 0.0),
        layerscale_init=kwargs.pop("layerscale_init", 1.0e-5),
        norm_layer=norm_layer,
        ffn_layer=kwargs.pop("ffn_layer", "mlp"),
        ffn_bias=kwargs.pop("ffn_bias", True),
        proj_bias=kwargs.pop("proj_bias", True),
        n_storage_tokens=kwargs.pop("n_storage_tokens", 4),
        mask_k_bias=kwargs.pop("mask_k_bias", True),
    )

    if kwargs.pop("pretrained", True):
        weights = kwargs.pop("weights", Weights.LVD1689M)
        hash_suffix = kwargs.pop("hash", None)
        kwargs.pop("version", None)  # accepted for compatibility but unused

        if isinstance(weights, Weights):
            url = _make_dinov3_vit_model_url(
                compact_arch_name="vits",
                patch_size=patch_size,
                weights=weights,
                hash_suffix=hash_suffix,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()

    return model


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    """Instantiate the ViT-S/16 backbone used by the few-shot framework."""

    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    return _load_vit(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        norm_layer="layernormbf16",
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )


__all__ = ["Weights", "convert_path_or_url_to_url", "dinov3_vits16"]
