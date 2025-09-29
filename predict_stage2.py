"""Run Stage-2 few-shot segmentation inference on POTSDAM_BIJIE.

Example usage (inside ``conda activate dl311_dino``)::

python predict_stage2.py \
    --run-dir runs/stage2_full/mySSL_stage2_full_POTSDAM_BIJIE_sets_8_shot/1 \
    --checkpoint runs/stage2_full/mySSL_stage2_full_POTSDAM_BIJIE_sets_8_shot/1/snapshots/6000.pth \
    --output-dir runs/stage2_full/mySSL_stage2_full_POTSDAM_BIJIE_sets_8_shot/1/predictions \
    --support-list data/potsdam_bijie/splits/support_ids.txt \
    --query-list data/potsdam_bijie/splits/test_ids.txt

Use ``--max-samples`` for smoke runs or ``--device cpu`` if CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from dataloaders.satellite_dataset import SatelliteFewShotDataset
from models.grid_proto_fewshot import FewShotSeg

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rasterio is required for exporting GeoTIFF predictions. "
        "Install it inside the dl311_dino environment."
    ) from exc


LOGGER = logging.getLogger("stage2_predict")


@dataclass
class InferenceConfig:
    """Minimal configuration needed for Stage-2 inference."""

    dataset_name: str
    data_root: Path
    input_size: int
    ignore_label: int
    class_id: int
    num_support: int
    npart: int
    model_cfg: Dict[str, object]
    val_wsize: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Sacred run directory (contains config.json and snapshots).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint (.pth) to load. Defaults to the latest file in run-dir/snapshots/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for predictions (default: <run-dir>/predictions).",
    )
    parser.add_argument(
        "--support-list",
        type=Path,
        default=None,
        help="Text file with support tile IDs (one per line). Overrides Sacred config list.",
    )
    parser.add_argument(
        "--query-list",
        type=Path,
        default=None,
        help="Text file with query/test tile IDs (one per line). Defaults to all tiles in test split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of tiles to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to 'cuda'.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():  # pragma: no cover
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_id_list(path: Path | None) -> List[str] | None:
    if path is None:
        return None
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(f"ID list not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if not ids:
        raise ValueError(f"ID list at {path} is empty")
    return ids


def build_inference_config(raw_cfg: Dict[str, object]) -> InferenceConfig:
    dataset_name = str(raw_cfg["dataset"])
    data_root = Path(raw_cfg["path"][dataset_name]["data_dir"]).expanduser()
    if not data_root.exists():  # pragma: no cover
        raise FileNotFoundError(f"Dataset root not found at {data_root}")

    act_labels = raw_cfg.get("dataset_act_labels", {}).get(dataset_name)
    if not act_labels:
        raise ValueError(f"Active class list missing for dataset '{dataset_name}' in config.json")
    class_id = int(act_labels[0])

    model_cfg = dict(raw_cfg["model"])
    adapter_path = model_cfg.get("adapter_state_path")
    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_file():
            LOGGER.warning("Adapter state path %s does not exist; continuing without it.", adapter_path)
            model_cfg["adapter_state_path"] = None

    return InferenceConfig(
        dataset_name=dataset_name,
        data_root=data_root,
        input_size=int(raw_cfg["input_size"][0]),
        ignore_label=int(raw_cfg.get("ignore_label", 255)),
        class_id=class_id,
        num_support=int(raw_cfg["task"]["n_shots"]),
        npart=int(raw_cfg["task"]["npart"]),
        model_cfg=model_cfg,
        val_wsize=int(raw_cfg.get("val_wsize")) if raw_cfg.get("val_wsize") is not None else None,
    )


def ensure_image_4d(tensor: Tensor) -> Tensor:
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    raise ValueError(f"Expected 3D or 4D tensor for image, got shape {tuple(tensor.shape)}")


def ensure_mask_3d(tensor: Tensor) -> Tensor:
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    raise ValueError(f"Expected 2D or 3D tensor for mask, got shape {tuple(tensor.shape)}")


def build_model(model_cfg: Dict[str, object], input_size: int, device: torch.device) -> FewShotSeg:
    model = FewShotSeg(image_size=input_size, cfg=model_cfg)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def load_checkpoint(model: FewShotSeg, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing %d keys when loading checkpoint: %s", len(missing), missing)
    if unexpected:
        LOGGER.warning("Unexpected %d keys when loading checkpoint: %s", len(unexpected), unexpected)


def resolve_checkpoint(run_dir: Path, explicit_ckpt: Path | None) -> Path:
    if explicit_ckpt is not None:
        return explicit_ckpt
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            "No checkpoint provided and snapshots directory not found under "
            f"{snapshots_dir}"
        )
    candidates = sorted(snapshots_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth files found under {snapshots_dir}")
    return candidates[-1]


def load_support_tensors(
    cfg: InferenceConfig,
    support_ids: Sequence[str],
    device: torch.device,
) -> tuple[list[list[Tensor]], list[list[Tensor]], list[list[Tensor]]]:
    dataset = SatelliteFewShotDataset(
        dataset_name=cfg.dataset_name,
        base_dir=cfg.data_root,
        split="train",
        transforms=None,
        scan_per_load=-1,
        image_size=cfg.input_size,
    )
    id_to_index = {scan_id: idx for idx, scan_id in enumerate(dataset.ids)}

    support_images: list[list[Tensor]] = [[]]
    support_fg_mask: list[list[Tensor]] = [[]]
    support_bg_mask: list[list[Tensor]] = [[]]

    for support_id in support_ids:
        if support_id not in id_to_index:
            raise KeyError(f"Support tile '{support_id}' not present in {cfg.dataset_name}/train")
        sample = dataset[id_to_index[support_id]]
        image = ensure_image_4d(sample["image"]).to(device=device, dtype=torch.float32)
        label = sample["label"]
        fg_mask = ensure_mask_3d((label == cfg.class_id).float()).to(device=device)
        bg_mask = ensure_mask_3d((label != cfg.class_id).float()).to(device=device)
        support_images[0].append(image)
        support_fg_mask[0].append(fg_mask)
        support_bg_mask[0].append(bg_mask)

    if not support_images[0]:
        raise ValueError("No support tiles were loaded")

    LOGGER.info("Loaded %d support tiles", len(support_images[0]))
    return support_images, support_fg_mask, support_bg_mask


def load_test_dataset(cfg: InferenceConfig) -> SatelliteFewShotDataset:
    return SatelliteFewShotDataset(
        dataset_name=cfg.dataset_name,
        base_dir=cfg.data_root,
        split="test",
        transforms=None,
        scan_per_load=-1,
        image_size=cfg.input_size,
    )


def compute_binary_iou(pred: np.ndarray, target: np.ndarray, ignore_label: int) -> float:
    mask = target != ignore_label
    if mask.sum() == 0:
        return float("nan")
    pred_fg = pred == 1
    target_fg = target == 1
    intersection = np.logical_and(pred_fg, target_fg)[mask].sum(dtype=np.float64)
    union = np.logical_or(pred_fg, target_fg)[mask].sum(dtype=np.float64)
    if union == 0:
        return float("nan")
    return float(intersection / union)


def save_geotiff(reference: Path, prediction: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference) as src:
        meta = src.meta.copy()
    meta.update({"count": 1, "dtype": prediction.dtype})
    with rasterio.open(destination, "w", **meta) as dst:
        dst.write(prediction, 1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    raw_cfg = load_config(args.run_dir)
    inf_cfg = build_inference_config(raw_cfg)

    support_ids = load_id_list(args.support_list) or list(raw_cfg.get("support_id_whitelist", []))
    if not support_ids:
        raise ValueError("Support tile list is empty. Provide --support-list or update config.json")
    if len(support_ids) < inf_cfg.num_support:
        LOGGER.warning(
            "Only %d support tiles supplied but training used %d shots; continuing anyway.",
            len(support_ids),
            inf_cfg.num_support,
        )

    query_ids = load_id_list(args.query_list)

    if args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    LOGGER.info("Using device %s", device)

    model = build_model(inf_cfg.model_cfg, inf_cfg.input_size, device)
    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    load_checkpoint(model, checkpoint_path, device)

    support_images, support_fg_mask, support_bg_mask = load_support_tensors(inf_cfg, support_ids, device)

    test_dataset = load_test_dataset(inf_cfg)
    id_to_index = {scan_id: idx for idx, scan_id in enumerate(test_dataset.ids)}

    if query_ids is None:
        ordered_ids = list(test_dataset.ids)
        LOGGER.info("No query list provided; predicting all %d test tiles.", len(ordered_ids))
    else:
        missing = [scan_id for scan_id in query_ids if scan_id not in id_to_index]
        if missing:
            raise KeyError(f"Query IDs not found in test split: {missing[:5]} ...")
        ordered_ids = list(query_ids)

    if args.max_samples is not None:
        ordered_ids = ordered_ids[: args.max_samples]
        LOGGER.info("Restricting inference to the first %d tiles", len(ordered_ids))

    output_dir = args.output_dir or (args.run_dir / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    geotiff_dir = output_dir / "geotiff"
    geotiff_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, float]] = []

    model.eval()
    with torch.no_grad():
        for scan_id in tqdm(ordered_ids, desc="Predicting", unit="tile"):
            sample = test_dataset[id_to_index[scan_id]]
            image = ensure_image_4d(sample["image"]).to(device=device, dtype=torch.float32)
            query_images = [image]

            logits, *_ = model(
                support_images,
                support_fg_mask,
                support_bg_mask,
                query_images,
                isval=True,
                val_wsize=inf_cfg.val_wsize,
            )

            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            target = sample["label"].cpu().numpy().astype(np.int64)
            iou = compute_binary_iou(prediction, target, inf_cfg.ignore_label)
            metrics.append({"scan_id": scan_id, "iou": iou})

            reference_path = inf_cfg.data_root / "img_dir" / "test" / f"{scan_id}.tif"
            if not reference_path.is_file():  # pragma: no cover
                LOGGER.warning("Skipping GeoTIFF export for %s (reference not found)", scan_id)
            else:
                save_geotiff(reference_path, prediction, geotiff_dir / f"{scan_id}.tif")

    ious = [m["iou"] for m in metrics if not np.isnan(m["iou"])]
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    summary = {
        "num_tiles": len(metrics),
        "mean_iou": mean_iou,
        "per_tile": metrics,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
