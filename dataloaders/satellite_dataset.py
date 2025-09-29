"""Satellite few-shot dataset utilities (rasterio backend)."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import cv2
import rasterio
import torch
from rasterio.errors import RasterioIOError
from tqdm.auto import tqdm

from dataloaders.common import BaseDataset, Subset
from dataloaders.dataset_utils import DATASET_INFO

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

logger = logging.getLogger(__name__)


class SatelliteFewShotDataset(BaseDataset):
    """Few-shot dataset wrapper for 2D remote sensing tiles."""

    def __init__(
        self,
        *,
        dataset_name: str,
        base_dir: str | Path,
        split: str,
        transforms,
        scan_per_load: int,
        image_size: int,
        support_id_whitelist: Optional[Iterable[str]] = None,
        query_id_whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(base_dir=str(base_dir))
        info = DATASET_INFO[dataset_name]

        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size

        class_id_map = info.get("CLASS_ID_MAP")
        if class_id_map is None:
            label_names = info["REAL_LABEL_NAME"]
            class_id_map = {name: idx for idx, name in enumerate(label_names)}
        max_class_id = max(class_id_map.values())
        self.label_name: List[str] = [""] * (max_class_id + 1)
        for name, cid in class_id_map.items():
            if cid >= len(self.label_name):
                self.label_name.extend([""] * (cid - len(self.label_name) + 1))
            self.label_name[cid] = name

        self.class_id_map = class_id_map
        self.background_id = info.get("BACKGROUND_ID", 0)
        self.forbidden_ids = set(info.get("FORBIDDEN_CLASS_IDS", []))
        self.nclass = len(self.label_name)
        self.transforms = transforms
        self.is_train = split == "train"
        self.scan_per_load = scan_per_load
        self.tile_z_dim = 1
        self.use_3_slices = False

        base_dir = Path(base_dir)
        img_root = base_dir / "img_dir" / split
        ann_root = base_dir / "ann_dir" / split
        if not img_root.exists() or not ann_root.exists():
            raise FileNotFoundError(
                f"Expecting image/label directories at {img_root} and {ann_root}."
            )

        self.samples: List[Dict[str, np.ndarray]] = []
        self.ids: List[str] = []
        self.support_whitelist = set(support_id_whitelist) if support_id_whitelist else None
        self.query_whitelist = set(query_id_whitelist) if query_id_whitelist else None
        self.scan_per_load = scan_per_load

        self.tile_records: List[Dict[str, Path]] = []
        for img_path in sorted(img_root.glob("*.tif")):
            ann_path = ann_root / img_path.name
            if not ann_path.exists():
                continue
            self.tile_records.append(
                {
                    "scan_id": img_path.stem,
                    "img_path": img_path,
                    "ann_path": ann_path,
                }
            )

        if not self.tile_records:
            raise RuntimeError(f"No samples found in {img_root}")

        self.scan_ids = [record["scan_id"] for record in self.tile_records]
        self._load_buffer()

    def _select_record_indices(self) -> List[int]:
        if self.scan_per_load is None or self.scan_per_load <= 0 or self.scan_per_load >= len(self.tile_records):
            return list(range(len(self.tile_records)))
        return random.sample(range(len(self.tile_records)), self.scan_per_load)

    def _load_buffer(self, record_indices: Optional[List[int]] = None) -> None:
        if record_indices is None:
            record_indices = self._select_record_indices()

        samples: List[Dict[str, np.ndarray]] = []
        ids: List[str] = []

        for idx in tqdm(
            record_indices,
            desc=f"Loading {self.dataset_name}:{self.split}",
            unit="tile",
            dynamic_ncols=True,
        ):
            record = self.tile_records[idx]
            img_path: Path = record["img_path"]
            ann_path: Path = record["ann_path"]
            sample_id: str = record["scan_id"]

            try:
                with rasterio.Env():
                    with rasterio.open(img_path) as src:
                        image = src.read(out_dtype="float32")
                        if src.count < 3:
                            raise ValueError(
                                f"Expected RGB imagery for {img_path}, got {src.count} channels"
                            )
                        image = np.transpose(image[:3, ...], (1, 2, 0)) / 255.0
            except (RasterioIOError, ValueError) as exc:
                logger.warning("Skipping image '%s': %s", img_path.name, exc)
                continue

            image = (image - IMAGENET_MEAN) / IMAGENET_STD

            try:
                with rasterio.Env():
                    with rasterio.open(ann_path) as src:
                        mask = src.read(1, out_dtype="uint8")
            except RasterioIOError as exc:
                logger.warning("Skipping annotation '%s': %s", ann_path.name, exc)
                continue

            mask = mask.astype(np.int64)
            target_size = (self.image_size, self.image_size)
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            if mask.shape != target_size:
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            if self.forbidden_ids and np.isin(mask, list(self.forbidden_ids)).any():
                continue

            samples.append({
                "image": image,
                "label": mask,
                "scan_id": sample_id,
            })
            ids.append(sample_id)

        if not samples:
            raise RuntimeError(
                f"Unable to load any samples for {self.dataset_name}:{self.split}. "
                "Check data integrity or reduce scan_per_load."
            )

        self.samples = samples
        self.ids = ids
        self.pid_curr_load = list(ids)

        query_whitelist = self.query_whitelist
        if query_whitelist is not None:
            active_indices = [idx for idx, scan_id in enumerate(self.ids) if scan_id in query_whitelist]
        else:
            active_indices = list(range(len(self.samples)))

        if not active_indices:
            raise RuntimeError(
                "No query tiles remain after applying whitelist/filtering. "
                "Check SUPPORT_TILE_FILE / STAGE2_TRAIN_QUERY_FILE settings."
            )

        self.active_indices = active_indices
        self.potential_support_sid = list(self.pid_curr_load)
        self.all_label_names = self.label_name
        foreground_ids = [cid for cid in range(len(self.label_name)) if cid != self.background_id]
        self.tp1_cls_map = {
            self.label_name[cid]: {scan_id: [0] for scan_id in self.pid_curr_load}
            for cid in foreground_ids
            if cid < len(self.label_name) and self.label_name[cid]
        }
        self._build_class_indices()

    def _build_class_indices(self) -> None:
        self.idx_by_class = {cid: [] for cid in range(len(self.label_name))}
        for local_idx, global_idx in enumerate(self.active_indices):
            mask = self.samples[global_idx]["label"]
            present_ids = np.unique(mask)
            for cid in present_ids:
                if cid >= len(self.label_name):
                    continue
                self.idx_by_class[cid].append(local_idx)

    def reload_buffer(self) -> None:
        if self.scan_per_load is None or self.scan_per_load <= 0 or self.scan_per_load >= len(self.tile_records):
            logger.info("reload_buffer skipped: loading full dataset in memory.")
            return

        self._load_buffer()

    def subsets(self, sub_args_lst=None):
        subsets = []
        for cid in range(len(self.label_name)):
            indices = self.idx_by_class.get(cid, [])
            if sub_args_lst is not None:
                subsets.append(Subset(dataset=self, indices=indices, sub_attrib_args=sub_args_lst[cid]))
            else:
                subsets.append(Subset(dataset=self, indices=indices))
        return subsets

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        global_idx = self.active_indices[index % len(self.active_indices)]
        record = self.samples[global_idx]
        image = record["image"]
        mask = record["label"]

        if self.transforms is not None and self.is_train:
            comp = np.concatenate([image, mask[..., None]], axis=-1)
            image, mask = self.transforms(
                comp,
                c_img=image.shape[-1],
                c_label=1,
                nclass=self.nclass,
                use_onehot=False,
            )
        else:
            mask = mask[..., None]

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        label = torch.from_numpy(mask.squeeze(-1)).long()

        sample = {
            "image": image,
            "label": label,
            "is_start": True,
            "is_end": True,
            "nframe": 1,
            "scan_id": record["scan_id"],
            "z_id": 0,
        }

        if self.aux_attrib:
            for key_prefix in self.aux_attrib:
                aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                for key_suffix, value in aux_attrib_val.items():
                    sample[f"{key_prefix}_{key_suffix}"] = value

        return sample

    def get_support(
        self,
        *,
        curr_class: int,
        class_idx: List[int],
        scan_idx: List[int],
        npart: int,
    ) -> Dict[str, List[List[torch.Tensor]]]:
        assert curr_class != 0
        support_images: List[List[torch.Tensor]] = [[]]
        support_mask: List[List[Dict[str, torch.Tensor]]] = [[]]

        pool = list(self.support_whitelist) if self.support_whitelist else self.pid_curr_load

        for sid in scan_idx:
            if not pool:
                continue
            idx = sid % len(pool)
            scan_id = pool[idx]
            global_idx = self.ids.index(scan_id)
            image = self.samples[global_idx]["image"]
            mask = self.samples[global_idx]["label"]

            img_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(mask.astype(np.int64))

            support_images[0].append(img_tensor)
            support_mask[0].append(
                {
                    "fg_mask": (mask_tensor == curr_class).unsqueeze(0).float(),
                    "bg_mask": (mask_tensor != curr_class).unsqueeze(0).float(),
                }
            )

        self.potential_support_sid = pool

        return {
            "support_images": support_images,
            "support_mask": support_mask,
            "support_inst": [[]],
            "support_scribbles": [[]],
        }
