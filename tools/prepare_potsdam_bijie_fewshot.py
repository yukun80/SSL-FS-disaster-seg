"""Utility script to curate few-shot splits for POTSDAM_BIJIE.

Usage example::

python -m tools.prepare_potsdam_bijie_fewshot \
--dataset-root data/potsdam_bijie \
--support-count 8 \
--output-dir data/potsdam_bijie/splits

The script scans the raster tiles, keeps track of tiles containing landslide
foreground, and writes plain-text lists that Stage-2 few-shot training can
reuse (support, query, val, test).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import rasterio

from dataloaders.dataset_utils import DATASET_INFO


@dataclass
class TileStat:
    stem: str
    pixels: int
    positives: int

    @property
    def has_positive(self) -> bool:
        return self.positives > 0

    @property
    def positive_ratio(self) -> float:
        return float(self.positives) / float(self.pixels) if self.pixels else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare few-shot splits for POTSDAM_BIJIE")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to data/potsdam_bijie")
    parser.add_argument(
        "--support-count",
        type=int,
        default=8,
        help="Number of tiles to reserve for the few-shot support set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the split files (defaults to <dataset-root>/splits)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="LANDSLIDE",
        help="Foreground class name to enforce in the support tiles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files",
    )
    return parser.parse_args()


def _collect_tiles(base: Path, split: str) -> List[Path]:
    img_dir = base / "img_dir" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing {img_dir}. Ensure the dataset follows img_dir/<split>/tile.tif")
    return sorted(img_dir.glob("*.tif"))


def _compute_tile_stats(tiles: Iterable[Path], mask_dir: Path, positive_id: int) -> List[TileStat]:
    stats: List[TileStat] = []
    for tile_path in tiles:
        mask_path = mask_dir / tile_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing annotation for {tile_path.name}")
        with rasterio.Env(), rasterio.open(mask_path) as src:
            mask = src.read(1, out_dtype="uint16")
        positives = int(np.count_nonzero(mask == positive_id))
        total = mask.size
        stats.append(TileStat(tile_path.stem, total, positives))
    return stats


def _resolve_positive_id(class_name: str) -> int:
    info = DATASET_INFO["POTSDAM_BIJIE"]
    mapping = info.get("CLASS_ID_MAP", {})
    if class_name not in mapping:
        raise KeyError(f"Class '{class_name}' not recognised. Available: {sorted(mapping)}")
    return mapping[class_name]


def _write_list(path: Path, stems: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for stem in stems:
            handle.write(f"{stem}\n")


def main() -> None:
    args = _parse_args()
    dataset_root: Path = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir or (dataset_root / "splits")

    positive_id = _resolve_positive_id(args.class_name)

    train_tiles = _collect_tiles(dataset_root, "train")
    val_tiles = _collect_tiles(dataset_root, "val")
    test_tiles = _collect_tiles(dataset_root, "test")

    train_stats = _compute_tile_stats(train_tiles, dataset_root / "ann_dir" / "train", positive_id)

    support = [stat for stat in train_stats if stat.has_positive]
    if len(support) < args.support_count:
        raise RuntimeError(
            f"Requested {args.support_count} support tiles but only {len(support)} contain class id {positive_id}."
        )
    support.sort(key=lambda item: item.positive_ratio, reverse=True)
    chosen_support = support[: args.support_count]
    chosen_support_stems = {stat.stem for stat in chosen_support}
    train_query_stems = [stat.stem for stat in train_stats if stat.stem not in chosen_support_stems]

    summary = {
        "support_tiles": len(chosen_support_stems),
        "train_query_tiles": len(train_query_stems),
        "val_tiles": len(val_tiles),
        "test_tiles": len(test_tiles),
        "support_positive_ratio_mean": (
            float(np.mean([s.positive_ratio for s in chosen_support])) if chosen_support else 0.0
        ),
    }

    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return

    _write_list(output_dir / "support_ids.txt", sorted(chosen_support_stems))
    _write_list(output_dir / "train_query_ids.txt", train_query_stems)
    _write_list(output_dir / "val_ids.txt", [path.stem for path in sorted(val_tiles)])
    _write_list(output_dir / "test_ids.txt", [path.stem for path in sorted(test_tiles)])


if __name__ == "__main__":
    main()
