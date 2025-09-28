"""Minimal UPerNet implementation tailored for ViT feature maps."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bins: Iterable[int], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for bin_size in bins
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + len(bins) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled: List[torch.Tensor] = [x]
        for block in self.blocks:
            pooled.append(F.interpolate(block(x), size=(h, w), mode="bilinear", align_corners=False))
        concatenated = torch.cat(pooled, dim=1)
        return self.project(concatenated)


class UPerNetHead(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        channels: int,
        ppm_bins: Iterable[int],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_channels = list(in_channels)
        assert len(in_channels) == 4, "UPerNetHead expects four feature maps."
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=False) for in_ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)) for _ in range(3)]
        )
        self.ppm = PyramidPoolingModule(channels, channels, ppm_bins, dropout)
        self.fpn_last = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [features[key] for key in ["res2", "res3", "res4", "res5"]]
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]
        laterals[-1] = self.ppm(laterals[-1])
        for idx in range(2, -1, -1):
            size = laterals[idx].shape[-2:]
            laterals[idx] = laterals[idx] + F.interpolate(laterals[idx + 1], size=size, mode="bilinear", align_corners=False)
            laterals[idx] = self.fpn_convs[idx](laterals[idx])
        laterals[-1] = self.fpn_last(laterals[-1])
        highest_res = laterals[0].shape[-2:]
        upsampled = [F.interpolate(feat, size=highest_res, mode="bilinear", align_corners=False) for feat in laterals]
        fused = torch.cat(upsampled, dim=1)
        bottleneck = self.fpn_bottleneck(fused)
        logits = self.classifier(bottleneck)
        return logits
