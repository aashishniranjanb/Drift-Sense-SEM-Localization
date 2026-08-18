"""
PACE Group Dataset Generator & Safe Patch Extraction Utilities
Mines Top-20 candidates per search image + 4 directional process-variation overlaps.
"""

import os
import csv
import argparse
import numpy as np
import cv2
import pandas as pd


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    mean, std = np.mean(img_f), np.std(img_f)
    if std > 1e-6:
        norm = (img_f - mean) / std
    else:
        norm = img_f - mean
    return norm.astype(np.float32)


def extract_patch_safe(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Safely extracts a patch of size x size centered at (cx, cy) with zero-padding near boundaries."""
    h, w = image.shape[:2]
    half = size / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + size
    y2 = y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    gx1, gx2 = max(0, x1), min(w, x2)
    gy1, gy2 = max(0, y1), min(h, y2)

    cropped = image[gy1:gy2, gx1:gx2]

    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return np.zeros((size, size), dtype=image.dtype)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

    if cropped.shape[:2] != (size, size):
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    return cropped


def extract_directional_overlaps(image: np.ndarray, cx: float, cy: float, patch_size: int = 32, offset: float = 40.0) -> np.ndarray:
    """
    Extracts 4 directional process-variation overlap patches (Top, Bottom, Left, Right).
    Returns (4, patch_size, patch_size) array.
    """
    top = extract_patch_safe(image, cx, cy - offset, patch_size)
    bottom = extract_patch_safe(image, cx, cy + offset, patch_size)
    left = extract_patch_safe(image, cx - offset, cy, patch_size)
    right = extract_patch_safe(image, cx + offset, cy, patch_size)

    top_norm = normalize_intensity(top)
    bottom_norm = normalize_intensity(bottom)
    left_norm = normalize_intensity(left)
    right_norm = normalize_intensity(right)

    return np.stack([top_norm, bottom_norm, left_norm, right_norm], axis=0).astype(np.float32)
