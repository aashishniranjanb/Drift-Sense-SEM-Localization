"""
V7 Feature Representations: Orientation Energy & High-Pass Structural Maps
Computes:
  1. High-Pass Structural Map (F_H = I - Gaussian(I))
  2. Structure Tensor Orientation Energy Map (F_O)
"""

import cv2
import numpy as np


def compute_highpass_map(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """High-pass structural map suppressing low-frequency SEM illumination and charging gradients."""
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    hp = img_f - blur
    hp = np.clip(hp + 0.5, 0.0, 1.0)
    return hp.astype(np.float32)


def compute_orientation_energy(image: np.ndarray) -> np.ndarray:
    """Computes local structure tensor orientation-consistency map (F_O)."""
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    j_xx = cv2.GaussianBlur(gx * gx, (5, 5), 1.0)
    j_yy = cv2.GaussianBlur(gy * gy, (5, 5), 1.0)
    j_xy = cv2.GaussianBlur(gx * gy, (5, 5), 1.0)

    # Local coherence / orientation strength
    trace = j_xx + j_yy + 1e-7
    det = j_xx * j_yy - j_xy * j_xy
    diff = j_xx - j_yy

    # Coherence metric = sqrt((j_xx - j_yy)^2 + 4 * j_xy^2) / trace
    coherence = np.sqrt(diff * diff + 4.0 * j_xy * j_xy) / trace
    coherence = np.clip(coherence, 0.0, 1.0)
    return coherence.astype(np.float32)
