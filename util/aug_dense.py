"""Augmentation utilities for dense remote-sensing segmentation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass
class AugmentationConfig:
    crop_size: int
    min_scale: float
    max_scale: float
    min_aspect: float
    max_aspect: float
    rotation_deg: float
    color_jitter: Tuple[float, float, float, float]
    cutmix_prob: float
    copy_paste_prob: float
    mean: Iterable[float]
    std: Iterable[float]
    ignore_index: int


def _sample_scale(min_scale: float, max_scale: float, min_aspect: float, max_aspect: float) -> Tuple[float, float]:
    scale = random.uniform(min_scale, max_scale)
    aspect = random.uniform(min_aspect, max_aspect)
    scale_x = scale * math.sqrt(aspect)
    scale_y = scale / math.sqrt(aspect)
    return scale_x, scale_y


def _resize(image: np.ndarray, mask: np.ndarray, scale_x: float, scale_y: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    new_h = max(int(height * scale_y), 1)
    new_w = max(int(width * scale_x), 1)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def _rotate(image: np.ndarray, mask: np.ndarray, angle: float, ignore_index: int) -> Tuple[np.ndarray, np.ndarray]:
    if abs(angle) < 1e-2:
        return image, mask
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    image_rot = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_rot = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=ignore_index,
    )
    return image_rot, mask_rot


def _pad_if_needed(image: np.ndarray, mask: np.ndarray, crop_size: int, ignore_index: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_h = max(crop_size - height, 0)
    pad_w = max(crop_size - width, 0)
    if pad_h <= 0 and pad_w <= 0:
        return image, mask
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=ignore_index)
    return image, mask


def _random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if height == crop_size and width == crop_size:
        return image, mask
    max_y = height - crop_size
    max_x = width - crop_size
    top = 0 if max_y <= 0 else random.randint(0, max_y)
    left = 0 if max_x <= 0 else random.randint(0, max_x)
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def _apply_color_jitter(image: np.ndarray, jitter: Tuple[float, float, float, float]) -> np.ndarray:
    brightness, contrast, saturation, hue = jitter
    if brightness > 0:
        alpha = 1.0 + random.uniform(-brightness, brightness)
        image = image * alpha
    if contrast > 0:
        mean = image.mean(axis=(0, 1), keepdims=True)
        alpha = 1.0 + random.uniform(-contrast, contrast)
        image = (image - mean) * alpha + mean
    if saturation > 0:
        gray = np.dot(image, np.array([0.114, 0.587, 0.299], dtype=image.dtype))[:, :, None]
        alpha = 1.0 + random.uniform(-saturation, saturation)
        image = (image - gray) * alpha + gray
    if hue > 0:
        hsv = cv2.cvtColor(np.clip(image, 0.0, 1.0), cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue) * 180.0) % 180.0
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(image, 0.0, 1.0)


def _maybe_horizontal_flip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1, :])
        mask = np.ascontiguousarray(mask[:, ::-1])
    return image, mask


def _cutmix(base_image: np.ndarray, base_mask: np.ndarray, mix_image: np.ndarray, mix_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = base_mask.shape
    cut_ratio = random.uniform(0.25, 0.5)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    base_image[y1:y2, x1:x2] = mix_image[y1:y2, x1:x2]
    base_mask[y1:y2, x1:x2] = mix_mask[y1:y2, x1:x2]
    return base_image, base_mask


def _copy_paste(base_image: np.ndarray, base_mask: np.ndarray, mix_image: np.ndarray, mix_mask: np.ndarray, ignore_index: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_classes = [c for c in np.unique(mix_mask) if c not in (ignore_index, 0)]
    if not unique_classes:
        return base_image, base_mask
    chosen_class = random.choice(unique_classes)
    binary_mask = (mix_mask == chosen_class).astype(np.uint8)
    if binary_mask.sum() == 0:
        return base_image, base_mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return base_image, base_mask
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    roi_mask = binary_mask[y : y + h, x : x + w]
    roi_image = mix_image[y : y + h, x : x + w]
    dest_x = random.randint(0, base_mask.shape[1] - w)
    dest_y = random.randint(0, base_mask.shape[0] - h)
    mask_slice = base_mask[dest_y : dest_y + h, dest_x : dest_x + w]
    image_slice = base_image[dest_y : dest_y + h, dest_x : dest_x + w]
    keep = roi_mask.astype(bool)
    image_slice[keep] = roi_image[keep]
    mask_slice[keep] = chosen_class
    base_image[dest_y : dest_y + h, dest_x : dest_x + w] = image_slice
    base_mask[dest_y : dest_y + h, dest_x : dest_x + w] = mask_slice
    return base_image, base_mask


class DenseAugmentations:
    """Callable augmentation pipeline used by the dense trainer."""

    def __init__(self, cfg: AugmentationConfig) -> None:
        self.cfg = cfg

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        scale_x, scale_y = _sample_scale(cfg.min_scale, cfg.max_scale, cfg.min_aspect, cfg.max_aspect)
        image, mask = _resize(image, mask, scale_x, scale_y)
        angle = random.uniform(-cfg.rotation_deg, cfg.rotation_deg)
        image, mask = _rotate(image, mask, angle, cfg.ignore_index)
        image, mask = _pad_if_needed(image, mask, cfg.crop_size, cfg.ignore_index)
        image, mask = _random_crop(image, mask, cfg.crop_size)
        image, mask = _maybe_horizontal_flip(image, mask)
        image = _apply_color_jitter(image, cfg.color_jitter)
        return image, mask

    def post_mix(
        self,
        base_image: np.ndarray,
        base_mask: np.ndarray,
        ref_image: np.ndarray,
        ref_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.cfg.cutmix_prob:
            base_image, base_mask = _cutmix(base_image, base_mask, ref_image, ref_mask)
        if random.random() < self.cfg.copy_paste_prob:
            base_image, base_mask = _copy_paste(base_image, base_mask, ref_image, ref_mask, self.cfg.ignore_index)
        return base_image, base_mask


def build_augmentation(config_dict: dict, ignore_index: int) -> DenseAugmentations:
    color_cfg = config_dict.get("color_jitter", {})
    jitter = (
        float(color_cfg.get("brightness", 0.0)),
        float(color_cfg.get("contrast", 0.0)),
        float(color_cfg.get("saturation", 0.0)),
        float(color_cfg.get("hue", 0.0)),
    )
    aug_cfg = AugmentationConfig(
        crop_size=int(config_dict["crop_size"]),
        min_scale=float(config_dict.get("min_scale", 1.0)),
        max_scale=float(config_dict.get("max_scale", 1.0)),
        min_aspect=float(config_dict.get("min_aspect", 1.0)),
        max_aspect=float(config_dict.get("max_aspect", 1.0)),
        rotation_deg=float(config_dict.get("rotation_deg", 0.0)),
        color_jitter=jitter,
        cutmix_prob=float(config_dict.get("cutmix_prob", 0.0)),
        copy_paste_prob=float(config_dict.get("copy_paste_prob", 0.0)),
        mean=tuple(config_dict.get("normalization", {}).get("mean", (0.0, 0.0, 0.0))),
        std=tuple(config_dict.get("normalization", {}).get("std", (1.0, 1.0, 1.0))),
        ignore_index=ignore_index,
    )
    return DenseAugmentations(aug_cfg)
