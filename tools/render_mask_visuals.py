#!/usr/bin/env python3
"""Convert integer segmentation masks into colored PNG previews.

Typical usage (after ``conda activate dl311_dino``)::

    python tools/render_mask_visuals.py \
        --input-dir runs/stage2_full/.../predictions/geotiff \
        --output-dir runs/stage2_full/.../predictions/mask_png

The tool scans all `.tif` files in the input directory, infers the present
class IDs, maps them to a simple color palette (``0``→black, others assigned
distinct vivid colors), and writes PNG masks plus a `legend.json` describing
the mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rasterio is required for reading GeoTIFF prediction masks. "
        "Install it in the dl311_dino environment."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required for exporting colored mask PNGs. "
        "Install it in the dl311_dino environment."
    ) from exc


LOGGER = logging.getLogger("render_mask_visuals")

# Distinct RGB colors for non-background classes (beyond index 0).
COLOR_TABLE: List[tuple[int, int, int]] = [
    (220, 20, 60),   # crimson
    (65, 105, 225),  # royal blue
    (255, 140, 0),   # dark orange
    (34, 139, 34),   # forest green
    (255, 215, 0),   # gold
    (138, 43, 226),  # blue violet
    (255, 105, 180), # hot pink
    (0, 206, 209),   # dark turquoise
    (139, 69, 19),   # saddle brown
    (176, 196, 222), # light steel blue
    (46, 139, 87),   # sea green
    (255, 99, 71),   # tomato
    (123, 104, 238), # medium slate blue
    (210, 105, 30),  # chocolate
    (70, 130, 180),  # steel blue
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing integer mask GeoTIFFs (e.g. Stage-2 predictions).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where colored PNG masks and legend.json will be stored.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for mask files inside the input directory (default: '*.tif').",
    )
    parser.add_argument(
        "--background-color",
        default="0,0,0",
        help="RGB triple for background class 0 (default: '0,0,0').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (per-file class stats).",
    )
    return parser.parse_args()


def list_mask_files(directory: Path, pattern: str) -> List[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' under {directory}")
    return files


def parse_rgb_triplet(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Background color must be an 'R,G,B' triplet")
    if any(not (0 <= v <= 255) for v in parts):
        raise ValueError("RGB components must be in [0, 255]")
    return tuple(parts)  # type: ignore[return-value]


def collect_class_ids(paths: Sequence[Path]) -> List[int]:
    class_ids: set[int] = set()
    for path in paths:
        with rasterio.open(path) as src:
            mask = src.read(1)
        uniques = np.unique(mask)
        class_ids.update(int(val) for val in uniques)
    return sorted(class_ids)


def build_color_map(class_ids: Sequence[int], background_color: tuple[int, int, int]) -> Dict[int, tuple[int, int, int]]:
    if not class_ids:
        raise ValueError("No class IDs found in the provided masks")
    color_map: Dict[int, tuple[int, int, int]] = {}
    for idx, class_id in enumerate(class_ids):
        if class_id == 0:
            color_map[class_id] = background_color
        else:
            palette_idx = (idx - 1) % len(COLOR_TABLE)
            color_map[class_id] = COLOR_TABLE[palette_idx]
    return color_map


def render_mask(mask: np.ndarray, color_map: Dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        mask_region = mask == class_id
        if not np.any(mask_region):
            continue
        rgb[mask_region] = color
    return rgb


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_mask_files(input_dir, args.pattern)
    background_color = parse_rgb_triplet(args.background_color)

    LOGGER.info("Scanning %d mask files under %s", len(files), input_dir)
    class_ids = collect_class_ids(files)
    LOGGER.info("Detected class IDs: %s", class_ids)

    color_map = build_color_map(class_ids, background_color)
    legend_path = output_dir / "legend.json"
    with legend_path.open("w", encoding="utf-8") as handle:
        json.dump({str(k): list(v) for k, v in color_map.items()}, handle, indent=2)
    LOGGER.info("Saved color legend to %s", legend_path)

    for path in files:
        with rasterio.open(path) as src:
            mask = src.read(1)
        uniques = np.unique(mask)
        if args.verbose:
            LOGGER.debug("%s -> classes %s", path.name, uniques.tolist())
        rgb = render_mask(mask, color_map)
        out_path = output_dir / f"{path.stem}.png"
        Image.fromarray(rgb).save(out_path)
    LOGGER.info("Finished writing %d PNG masks to %s", len(files), output_dir)


if __name__ == "__main__":
    main()
