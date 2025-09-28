"""Backbone wrapper that adds channel adapters and LoRA to DINOv3."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from dinov3.hub.backbones import dinov3_vits16
from util.lora import inject_trainable_lora


LOGGER = logging.getLogger(__name__)


@dataclass
class BackboneConfig:
    in_channels: int
    adapter_channels: int
    freeze_backbone_at: int
    lora_rank: int
    target_layers: Sequence[int]
    weights: str | None = None


def _init_channel_adapter(conv: nn.Conv2d, in_channels: int) -> None:
    with torch.no_grad():
        nn.init.zeros_(conv.bias)
        nn.init.zeros_(conv.weight)
        channels = min(in_channels, conv.out_channels)
        for idx in range(channels):
            conv.weight[idx, idx % in_channels, 0, 0] = 1.0


class Dinov3BackboneWithAdapter(nn.Module):
    """Compose DINOv3 with a learnable channel adapter and LoRA."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        weights_arg = Path(cfg.weights).expanduser() if cfg.weights else None
        if weights_arg and weights_arg.exists():
            self.backbone = dinov3_vits16(pretrained=True, weights=str(weights_arg))
        else:
            if weights_arg:
                LOGGER.warning(
                    "Requested DINOv3 weights at %s were not found. Initialising the backbone randomly.",
                    weights_arg,
                )
            self.backbone = dinov3_vits16(pretrained=False)
        self.backbone.eval()
        self.channel_adapter = nn.Conv2d(cfg.in_channels, cfg.adapter_channels, kernel_size=1)
        _init_channel_adapter(self.channel_adapter, cfg.in_channels)
        self.adapter_norm = nn.BatchNorm2d(cfg.adapter_channels)
        self.feature_norm = nn.GroupNorm(1, self.backbone.embed_dim)
        self.target_layers = list(cfg.target_layers)
        self.lora_params: List[nn.Parameter] = []
        self.backbone_params: List[nn.Parameter] = []
        self._setup_trainable_parameters(cfg.freeze_backbone_at, cfg.lora_rank)

    def _setup_trainable_parameters(self, freeze_at: int, lora_rank: int) -> None:
        num_blocks = len(self.backbone.blocks)
        if freeze_at >= num_blocks and freeze_at >= 0:
            LOGGER.warning(
                "freeze_backbone_at=%s exceeds backbone depth (%s). All transformer blocks will be frozen.",
                freeze_at,
                num_blocks,
            )

        invalid_layers = [layer for layer in self.target_layers if layer >= num_blocks]
        if invalid_layers:
            LOGGER.warning(
                "Requested target layers %s exceed backbone depth (%s). These layers will be ignored.",
                invalid_layers,
                num_blocks,
            )
            self.target_layers = [layer for layer in self.target_layers if layer < num_blocks]

        def register_param(param: nn.Parameter, trainable: bool) -> None:
            if not isinstance(param, nn.Parameter):
                return
            param.requires_grad_(trainable)
            if trainable:
                self.backbone_params.append(param)

        for idx, block in enumerate(self.backbone.blocks):
            trainable = idx >= freeze_at if freeze_at >= 0 else True
            for param in block.parameters():
                param.requires_grad_(trainable)
                if trainable:
                    self.backbone_params.append(param)
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            self.backbone_params.append(param)
        trainable_tokens = freeze_at < 0 or freeze_at < num_blocks
        register_param(getattr(self.backbone, "cls_token", None), trainable_tokens)
        register_param(getattr(self.backbone, "mask_token", None), trainable_tokens)
        storage_tokens = getattr(self.backbone, "storage_tokens", None)
        if isinstance(storage_tokens, nn.Parameter):
            register_param(storage_tokens, trainable_tokens)
        elif isinstance(storage_tokens, nn.Module):
            for param in storage_tokens.parameters():
                register_param(param, trainable_tokens)
        if lora_rank > 0:
            groups, _ = inject_trainable_lora(
                self.backbone,
                target_replace_module={"SelfAttention"},
                r=lora_rank,
                dropout_p=0.0,
                scale=1.0,
            )
            for group in groups:
                params = list(group)
                for param in params:
                    param.requires_grad = True
                    self.lora_params.append(param)
        self.channel_adapter.requires_grad_(True)
        self.adapter_norm.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.channel_adapter(images)
        x = self.adapter_norm(x)
        layer_outputs = self.backbone.get_intermediate_layers(
            x,
            n=self.target_layers,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        keys = ["res2", "res3", "res4", "res5"]
        features: Dict[str, torch.Tensor] = {}
        for key, feat in zip(keys, layer_outputs):
            features[key] = self.feature_norm(feat)
        return features

    def param_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "adapter": list(self.channel_adapter.parameters()) + list(self.adapter_norm.parameters()),
            "lora": self.lora_params,
            "backbone": self.backbone_params,
        }
