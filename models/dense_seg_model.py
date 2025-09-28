"""Dense supervised segmentation model built on top of DINOv3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.dinov3_adapter import BackboneConfig, Dinov3BackboneWithAdapter
from .upernet import UPerNetHead


@dataclass
class DenseModelConfig:
    backbone: BackboneConfig
    head_channels: int
    ppm_bins: Iterable[int]
    dropout: float
    num_classes: int


class DenseSegmentationModel(nn.Module):
    def __init__(self, cfg: DenseModelConfig) -> None:
        super().__init__()
        self.backbone = Dinov3BackboneWithAdapter(cfg.backbone)
        embed_dim = self.backbone.backbone.embed_dim
        self.decode_head = UPerNetHead(
            in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
            channels=cfg.head_channels,
            ppm_bins=cfg.ppm_bins,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.decode_head.parameters()

    def param_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups = self.backbone.param_groups()
        groups["head"] = list(self.decode_head.parameters())
        return groups

    def export_adapter_state(self) -> Dict[str, torch.Tensor]:
        state = {
            "channel_adapter": self.backbone.channel_adapter.state_dict(),
            "adapter_norm": self.backbone.adapter_norm.state_dict(),
        }
        lora_state = {}
        for name, module in self.backbone.backbone.named_modules():
            if hasattr(module, "lora_up") and hasattr(module, "lora_down"):
                lora_state[f"{name}.lora_up"] = module.lora_up.state_dict()
                lora_state[f"{name}.lora_down"] = module.lora_down.state_dict()
        state["lora"] = lora_state
        return state
