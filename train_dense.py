"""Stage-1 dense supervised training entry point.

Launch example (after ``conda activate dl311_dino``)::

python train_dense.py --config configs/dense/openearthmap.yaml

"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from engine.dense_trainer import DenseTrainer


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense DINOv3 remote-sensing pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Override configuration. Format: section.key=value",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return config


def _set_by_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    target = obj
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    if not overrides:
        return config
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' is not in key=value format")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        _set_by_path(updated, key, value)
    return updated


def main() -> None:
    args = _parse_cli()
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} was not found")
    config = _load_config(config_path)
    config = _apply_overrides(config, args.opts)
    trainer = DenseTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
