"""Satellite few-shot dataset utilities (rasterio backend)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import rasterio
import torch

from dataloaders.common import BaseDataset, Subset
from dataloaders.dataset_utils import DATASET_INFO

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        support_id_whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(base_dir=str(base_dir))
        info = DATASET_INFO[dataset_name]

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

        for img_path in sorted(img_root.glob("*.tif")):
            ann_path = ann_root / img_path.name
            if not ann_path.exists():
                continue

            sample_id = img_path.stem

            with rasterio.Env():
                with rasterio.open(img_path) as src:
                    image = src.read(out_dtype="float32")
                    if src.count < 3:
                        raise ValueError(
                            f"Expected RGB imagery for {img_path}, got {src.count} channels"
                        )
                    image = np.transpose(image[:3, ...], (1, 2, 0)) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

            with rasterio.Env():
                with rasterio.open(ann_path) as src:
                    mask = src.read(1, out_dtype="uint8")

            mask = mask.astype(np.int64)
            if self.forbidden_ids and np.isin(mask, list(self.forbidden_ids)).any():
                continue

            self.samples.append(
                {
                    "image": image,
                    "label": mask,
                    "scan_id": sample_id,
                }
            )
            self.ids.append(sample_id)

        if not self.samples:
            raise RuntimeError(f"No samples found in {img_root}")

        self.scan_ids = list(self.ids)
        self.pid_curr_load = list(self.scan_ids)
        self.all_label_names = self.label_name
        foreground_ids = [cid for cid in range(len(self.label_name)) if cid != self.background_id]
        self.tp1_cls_map = {
            self.label_name[cid]: {scan_id: [0] for scan_id in self.scan_ids}
            for cid in foreground_ids
            if cid < len(self.label_name) and self.label_name[cid]
        }
        self.potential_support_sid: List[str] = []
        self.active_indices = list(range(len(self.samples)))
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
        if self.scan_per_load is None or self.scan_per_load <= 0 or self.scan_per_load >= len(self.scan_ids):
            self.active_indices = list(range(len(self.samples)))
        else:
            chosen = random.sample(self.scan_ids, k=self.scan_per_load)
            self.active_indices = [self.ids.index(pid) for pid in chosen]
        random.shuffle(self.active_indices)
        self._build_class_indices()

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
