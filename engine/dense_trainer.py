"""Training harness for dense DINOv3 segmentation."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from dataloaders.dense_dataset import build_dense_datasets
from models.backbones.dinov3_adapter import BackboneConfig
from models.dense_seg_model import DenseModelConfig, DenseSegmentationModel


@dataclass
class OptimConfig:
    epochs: int
    warmup_epochs: int
    base_lr: float
    min_lr: float
    weight_decay: float
    betas: Tuple[float, float]
    gradient_clip_norm: float
    lr_multipliers: Dict[str, float]


class WarmupCosineScheduler(LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float, base_lr: float) -> None:
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)
        self.min_lr_ratio = min_lr / base_lr
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        super().__init__(optimizer, lr_lambda)


def _create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_dataset = build_dense_datasets(dataset_cfg, dataset_cfg, is_train=True)
    val_dataset = build_dense_datasets(dataset_cfg, dataset_cfg, is_train=False)
    batch_size = int(dataset_cfg.get("batch_size", 4))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def _build_model(config: dict) -> DenseSegmentationModel:
    backbone_cfg = BackboneConfig(
        in_channels=int(config["dataset"].get("in_channels", 3)),
        adapter_channels=int(config["model"].get("adapter_channels", 3)),
        freeze_backbone_at=int(config["model"].get("freeze_backbone_at", 9)),
        lora_rank=int(config["model"].get("lora_rank", 8)),
        target_layers=config["model"].get("target_layers", [2, 5, 8, 11]),
        weights=config["model"].get("weights"),
    )
    model_cfg = DenseModelConfig(
        backbone=backbone_cfg,
        head_channels=int(config["model"].get("head_channels", 256)),
        ppm_bins=tuple(config["model"].get("ppm_bins", [1, 2, 3, 6])),
        dropout=float(config["model"].get("dropout", 0.1)),
        num_classes=int(config["dataset"].get("num_classes")),
    )
    return DenseSegmentationModel(model_cfg)


def _collect_param_groups(model: DenseSegmentationModel, optim_cfg: OptimConfig) -> Iterable[Dict[str, object]]:
    groups = model.param_groups()
    multipliers = optim_cfg.lr_multipliers
    param_groups = []
    for name in ["backbone", "adapter", "lora", "head"]:
        if name not in groups or not groups[name]:
            continue
        lr_scale = multipliers.get(name, 1.0)
        params = list(groups[name])
        if not params:
            continue
        param_groups.append({"params": params, "lr_scale": lr_scale, "name": name})
    return param_groups


def _build_optimizer(model: DenseSegmentationModel, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
    param_groups = _collect_param_groups(model, optim_cfg)
    optim_groups = [
        {
            "params": group["params"],
            "lr": optim_cfg.base_lr * group["lr_scale"],
            "weight_decay": optim_cfg.weight_decay,
            "name": group["name"],
        }
        for group in param_groups
    ]
    optimizer = AdamW(optim_groups, betas=optim_cfg.betas)
    return optimizer


def _compute_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]
        confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=logits.device)
        indices = target * num_classes + preds
        confmat.view(-1).index_add_(0, indices, torch.ones_like(indices, dtype=torch.long, device=logits.device))
        true_positive = confmat.diag()
        denominator = confmat.sum(dim=1) + confmat.sum(dim=0) - true_positive
        iou = true_positive.float() / denominator.clamp(min=1)
        return confmat, iou


class DenseTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optim_conf = config["optimization"]
        self.optim_cfg = OptimConfig(
            epochs=int(optim_conf["epochs"]),
            warmup_epochs=int(optim_conf.get("warmup_epochs", 0)),
            base_lr=float(optim_conf["base_lr"]),
            min_lr=float(optim_conf.get("min_lr", 1e-6)),
            weight_decay=float(optim_conf.get("weight_decay", 0.0)),
            betas=tuple(optim_conf.get("betas", [0.9, 0.999])),
            gradient_clip_norm=float(optim_conf.get("gradient_clip_norm", 0.0)),
            lr_multipliers=dict(optim_conf.get("lr_multipliers", {})),
        )
        self.output_dir = Path(config["logging"].get("output_dir", "runs/dense")).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(config["logging"].get("save_every", 5))
        self.val_every = int(config["logging"].get("val_every", 1))
        self.log_interval = int(config["logging"].get("log_interval", 20))
        self.ignore_index = int(config["dataset"].get("ignore_index", 255))
        torch.manual_seed(int(config["logging"].get("seed", 42)))

    def fit(self) -> None:
        train_loader, val_loader = _create_dataloaders(self.config)
        model = _build_model(self.config).to(self.device)
        optimizer = _build_optimizer(model, self.optim_cfg)
        total_steps = len(train_loader) * self.optim_cfg.epochs
        warmup_steps = len(train_loader) * self.optim_cfg.warmup_epochs
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, self.optim_cfg.min_lr, self.optim_cfg.base_lr)
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        start_epoch = 0
        resume_path = self.config["logging"].get("resume")
        if resume_path:
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
        global_step = start_epoch * len(train_loader)
        best_miou = 0.0
        for epoch in range(start_epoch, self.optim_cfg.epochs):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.optim_cfg.epochs}", leave=False)
            for step, batch in enumerate(progress, start=1):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                if self.optim_cfg.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.optim_cfg.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += loss.item()
                global_step += 1
                if step % self.log_interval == 0:
                    lr_info = {group.get("name", str(idx)): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
                    avg_loss = epoch_loss / step
                    print(f"Epoch {epoch+1:03d}/{self.optim_cfg.epochs:03d} | Step {step:04d}/{len(train_loader):04d} | Loss {avg_loss:.4f} | LR {lr_info}")
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} finished in {elapsed:.1f}s | loss={epoch_loss/len(train_loader):.4f}")
            if (epoch + 1) % self.val_every == 0:
                miou = self.validate(model, val_loader)
                if miou > best_miou:
                    best_miou = miou
                    self._save_checkpoint(model, optimizer, scheduler, scaler, epoch, best=True)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(model, optimizer, scheduler, scaler, epoch, best=False)
        self._save_adapter_only(model)

    def validate(self, model: DenseSegmentationModel, loader: DataLoader) -> float:
        model.eval()
        num_classes = int(self.config["dataset"]["num_classes"])
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                logits = model(images)
                confmat, _ = _compute_metrics(logits, masks, num_classes, self.ignore_index)
                confusion += confmat
        tp = confusion.diag().float()
        denom = confusion.sum(dim=1) + confusion.sum(dim=0) - tp
        iou = tp / denom.clamp(min=1.0)
        miou = iou.mean().item()
        print(f"Validation mIoU: {miou:.4f}")
        model.train()
        return miou

    def _save_checkpoint(self, model: DenseSegmentationModel, optimizer: torch.optim.Optimizer, scheduler: LambdaLR, scaler: GradScaler, epoch: int, best: bool) -> None:
        save_path = self.output_dir / ("best.pt" if best else f"epoch_{epoch+1:03d}.pt")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(payload, save_path)
        print(f"Checkpoint saved to {save_path}")

    def _save_adapter_only(self, model: DenseSegmentationModel) -> None:
        adapter_state = model.export_adapter_state()
        adapter_path = self.output_dir / "adapter_lora_state.pt"
        torch.save(adapter_state, adapter_path)
        print(f"Adapter + LoRA state exported to {adapter_path}")
