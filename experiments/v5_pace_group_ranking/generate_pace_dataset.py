"""
PACE Group Candidate Ranking Dataset Generator
Mines Top-20 candidate sets per search image for Group Candidate List Ranking training.

Each group sample contains:
- Reference 64x64, 128x128, 4x32x32 patches
- 20 Candidate 64x64, 128x128, 4x32x32 patches
- Ground truth target candidate index in [0, 19]
"""

import os
import sys
import csv
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))

from hf_space.src.pipeline import generate_sample, GenerationParams


DIFFICULTY_CONFIGS = {
    "Easy": GenerationParams(dose_search=500.0, shear_amplitude_px=0.5, drift_jitter_px=0.2, detector_noise_sigma_search=3.0),
    "Medium": GenerationParams(dose_search=200.0, shear_amplitude_px=1.5, drift_jitter_px=0.5, detector_noise_sigma_search=5.0),
    "Hard": GenerationParams(dose_search=100.0, shear_amplitude_px=3.0, drift_jitter_px=1.0, detector_noise_sigma_search=8.0, speckle_sigma=0.1, charging_streak_prob=1.0, charging_streak_intensity=1.0),
    "Adversarial": GenerationParams(dose_search=55.0, shear_amplitude_px=4.5, drift_jitter_px=2.0, detector_noise_sigma_search=12.0, speckle_sigma=0.25, salt_pepper_prob=0.008, charging_streak_prob=3.0, charging_streak_intensity=2.0),
}

ARCHITECTURES = ["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (1, 99))
    if p_high > p_low:
        return np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    return (img_f / 255.0).astype(np.float32)


def extract_gradient(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    mx = g.max()
    if mx > 1e-6:
        g /= mx
    return g.astype(np.float32)


def extract_patch_safe(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT)
        x1p = int(round(cx))
        y1p = int(round(cy))
        patch = padded[y1p:y1p+size, x1p:x1p+size]
    else:
        patch = image[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def extract_directional_overlaps(image: np.ndarray, cx: float, cy: float, patch_size: int = 32, offset: int = 40) -> np.ndarray:
    """Extracts 4 directional transition overlap patches (Top, Bottom, Left, Right). Returns (4, 32, 32)."""
    top = extract_patch_safe(image, cx, cy - offset, patch_size)
    bot = extract_patch_safe(image, cx, cy + offset, patch_size)
    left = extract_patch_safe(image, cx - offset, cy, patch_size)
    right = extract_patch_safe(image, cx + offset, cy, patch_size)

    top_norm = normalize_intensity(top)
    bot_norm = normalize_intensity(bot)
    left_norm = normalize_intensity(left)
    right_norm = normalize_intensity(right)

    return np.stack([top_norm, bot_norm, left_norm, right_norm], axis=0).astype(np.float32)


def mine_pace_group_sample(ref_img: np.ndarray, search_img: np.ndarray, gt_x: float, gt_y: float, top_k: int = 20) -> dict:
    """Runs FFT-NCC to extract Top-20 candidates and directional process overlap features."""
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    ref_norm = normalize_intensity(ref_100)
    ref_grad = extract_gradient(ref_100)
    search_norm = normalize_intensity(search_img)
    search_grad = extract_gradient(search_img)

    c_i = cv2.matchTemplate(search_norm, ref_norm, cv2.TM_CCOEFF_NORMED)
    c_g = cv2.matchTemplate(search_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    c_combo = 0.55 * c_i + 0.45 * c_g

    work = c_combo.copy()
    ch, cw = work.shape
    candidates = []

    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + 50.0
        cy = py + 50.0
        err = np.hypot(cx - gt_x, cy - gt_y)

        p64 = extract_patch_safe(search_img, cx, cy, 64)
        p128 = extract_patch_safe(search_img, cx, cy, 128)
        povl = extract_directional_overlaps(search_img, cx, cy, 32, offset=40)

        candidates.append({
            "cx": cx, "cy": cy,
            "ncc_score": float(max_val),
            "err_to_gt": float(err),
            "p64": normalize_intensity(p64),
            "p128": normalize_intensity(p128),
            "povl": povl
        })

        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0

    target_idx = -1
    for idx, c in enumerate(candidates):
        if c["err_to_gt"] <= 8.0:
            target_idx = idx
            break

    ref_64 = cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA)
    ref_128 = cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA)
    ref_ovl = extract_directional_overlaps(ref_img, 500, 500, 32, offset=200)

    return {
        "ref_64": normalize_intensity(ref_64),
        "ref_128": normalize_intensity(ref_128),
        "ref_ovl": ref_ovl,
        "candidates": candidates,
        "target_idx": target_idx
    }


def generate_pace_dataset(output_dir: str, num_samples: int = 500, seed_base: int = 70000):
    os.makedirs(os.path.join(output_dir, "groups"), exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    fieldnames = ["group_id", "difficulty", "architecture", "num_candidates", "target_idx", "group_file"]

    difficulties = list(DIFFICULTY_CONFIGS.keys())
    saved_count = 0
    t0 = time.time()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_idx in range(num_samples):
            seed = seed_base + sample_idx * 37
            rng = np.random.default_rng(seed)
            diff = difficulties[sample_idx % len(difficulties)]
            arch = ARCHITECTURES[sample_idx % len(ARCHITECTURES)]
            params = DIFFICULTY_CONFIGS[diff]

            try:
                sample = generate_sample(arch, rng, params)
            except Exception:
                continue

            ref_img = sample["reference_img"]
            search_img = sample["search_img"]
            gt_x, gt_y = float(sample["gt_x"]), float(sample["gt_y"])

            mined = mine_pace_group_sample(ref_img, search_img, gt_x, gt_y, top_k=20)

            if mined["target_idx"] < 0:
                continue  # Skip samples where true candidate didn't land in Top-20

            group_filename = f"group_{saved_count:05d}.npz"
            group_filepath = os.path.join(output_dir, "groups", group_filename)

            # Package group data arrays
            cands = mined["candidates"]
            cand_64 = np.stack([c["p64"] for c in cands], axis=0)      # (K, 64, 64)
            cand_128 = np.stack([c["p128"] for c in cands], axis=0)    # (K, 128, 128)
            cand_ovl = np.stack([c["povl"] for c in cands], axis=0)    # (K, 4, 32, 32)
            cand_ncc = np.array([c["ncc_score"] for c in cands], dtype=np.float32)

            np.savez_compressed(
                group_filepath,
                ref_64=mined["ref_64"],
                ref_128=mined["ref_128"],
                ref_ovl=mined["ref_ovl"],
                cand_64=cand_64,
                cand_128=cand_128,
                cand_ovl=cand_ovl,
                cand_ncc=cand_ncc,
                target_idx=mined["target_idx"]
            )

            writer.writerow({
                "group_id": saved_count,
                "difficulty": diff,
                "architecture": arch,
                "num_candidates": len(cands),
                "target_idx": mined["target_idx"],
                "group_file": group_filepath
            })

            saved_count += 1
            if saved_count % 50 == 0:
                print(f"  [{saved_count}/{num_samples}] PACE group candidate sets generated in {time.time()-t0:.1f}s")

    print(f"\nSuccessfully generated {saved_count} PACE group candidate sets in '{output_dir}'!")


if __name__ == "__main__":
    print("=" * 75)
    print("  Generating PACE Group Candidate List Ranking Dataset...")
    print("=" * 75)
    generate_pace_dataset("data/pace_train", num_samples=500, seed_base=70000)
