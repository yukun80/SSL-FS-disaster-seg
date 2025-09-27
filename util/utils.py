"""Utility helpers tailored to the remote-sensing few-shot pipeline."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compose_wt_simple(
    is_wce: bool,
    data_name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return class weights for cross-entropy.

    The remote landslide task is binary (background/foreground). When ``is_wce``
    is False we fall back to uniform weighting. Otherwise, we slightly down-weight
    the dominant background class. ``device`` defaults to CUDA when available so
    subsequent loss computations can run without manual transfers.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_weights = torch.tensor([0.05, 1.0], dtype=torch.float32, device=device)
    if not is_wce:
        return torch.ones_like(base_weights)
    return base_weights

