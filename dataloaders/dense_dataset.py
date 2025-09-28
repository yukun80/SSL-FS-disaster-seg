"""Dense supervised dataset utilities for remote-sensing segmentation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:  # pragma: no cover
    rasterio = None
    NotGeoreferencedWarning = None

from dataloaders.dataset_utils import DATASET_INFO
from util.aug_dense import DenseAugmentations, build_augmentation


@dataclass
class DatasetConfig:
    name: str
    root: Path
    split: str
    in_channels: int
    num_classes: int
    normalization_mean: Tuple[float, ...]
    normalization_std: Tuple[float, ...]
    ignore_index: int
    cache_images: bool = True


def _resolve_paths(root: Path, split: str) -> List[Tuple[Path, Path]]:
    img_root = root / "img_dir" / split
    ann_root = root / "ann_dir" / split
    if not img_root.exists() or not ann_root.exists():
        raise FileNotFoundError(f"Missing data folders: {img_root} or {ann_root}.")
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_root.glob("*.tif")):
        ann_path = ann_root / img_path.name
        if ann_path.exists():
            pairs.append((img_path, ann_path))
    if not pairs:
        raise RuntimeError(f"No tiles found in {img_root}.")
    return pairs


def _load_tile(img_path: Path, ann_path: Path, in_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    if rasterio is None:
        raise ImportError("rasterio is required to load remote-sensing tiles. Install rasterio before training.")
    warn_ctx = warnings.catch_warnings()
    if NotGeoreferencedWarning is not None:
        warn_ctx.__enter__()
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
    try:
        with rasterio.Env():
            with rasterio.open(img_path) as src:
                image = src.read(out_dtype="float32")
                if src.count < in_channels:
                    raise ValueError(f"Expected at least {in_channels} channels in {img_path}, found {src.count}.")
                image = np.transpose(image[:in_channels, ...], (1, 2, 0)) / 255.0
        with rasterio.Env():
            with rasterio.open(ann_path) as src:
                mask = src.read(1, out_dtype="uint8")
    finally:
        if NotGeoreferencedWarning is not None:
            warn_ctx.__exit__(None, None, None)
    return image.astype(np.float32, copy=False), mask.astype(np.int64, copy=False)


class DenseRemoteSensingDataset(Dataset):
    """Dataset yielding augmented tiles for dense supervised training."""

    def __init__(
        self,
        cfg: DatasetConfig,
        augment_dict: Optional[dict] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        if rasterio is None:
            raise ImportError("rasterio is required to use DenseRemoteSensingDataset")
        self.cfg = cfg
        self.is_train = is_train
        self.records = _resolve_paths(cfg.root, cfg.split)
        self.mean = np.asarray(cfg.normalization_mean, dtype=np.float32)
        self.std = np.asarray(cfg.normalization_std, dtype=np.float32)
        self.ignore_index = cfg.ignore_index
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.augmentor: Optional[DenseAugmentations]
        if is_train and augment_dict is not None:
            self.augmentor = build_augmentation(augment_dict, cfg.ignore_index)
        else:
            self.augmentor = None

    def __len__(self) -> int:
        return len(self.records)

    def _get_raw(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.cache_images and index in self.cache:
            image, mask = self.cache[index]
            return image.copy(), mask.copy()
        img_path, ann_path = self.records[index]
        image, mask = _load_tile(img_path, ann_path, self.cfg.in_channels)
        if self.cfg.cache_images:
            self.cache[index] = (image, mask)
        return image.copy(), mask.copy()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, mask = self._get_raw(index)
        if self.is_train and self.augmentor is not None:
            image, mask = self.augmentor(image, mask)
            if (self.augmentor.cfg.cutmix_prob > 0.0 or self.augmentor.cfg.copy_paste_prob > 0.0) and len(self.records) > 1:
                alt_index = random.randrange(len(self.records))
                if alt_index == index:
                    alt_index = (alt_index + 1) % len(self.records)
                ref_img, ref_mask = self._get_raw(alt_index)
                ref_img, ref_mask = self.augmentor(ref_img, ref_mask)
                image, mask = self.augmentor.post_mix(image, mask, ref_img, ref_mask)
        else:
            # Validation simply uses the original 512x512 tiles.
            pass
        image = np.clip(image, 0.0, 1.0)
        image = (image - self.mean) / self.std
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return {"image": image_tensor, "mask": mask_tensor}


def build_dense_datasets(config: dict, augment_config: dict, is_train: bool) -> DenseRemoteSensingDataset:
    dataset_name = config["name"]
    dataset_info = DATASET_INFO.get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    cfg = DatasetConfig(
        name=dataset_name,
        root=Path(config["root"]).expanduser(),
        split=config["train_split" if is_train else "val_split"],
        in_channels=int(config.get("in_channels", 3)),
        num_classes=int(config.get("num_classes", len(dataset_info["REAL_LABEL_NAME"]))),
        normalization_mean=tuple(config.get("normalization", {}).get("mean", (0.0, 0.0, 0.0))),
        normalization_std=tuple(config.get("normalization", {}).get("std", (1.0, 1.0, 1.0))),
        ignore_index=int(config.get("ignore_index", 255)),
        cache_images=bool(config.get("cache_images", True)),
    )
    return DenseRemoteSensingDataset(cfg, augment_config if is_train else None, is_train=is_train)
