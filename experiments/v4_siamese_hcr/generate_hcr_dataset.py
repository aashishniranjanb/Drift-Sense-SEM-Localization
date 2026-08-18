"""
Hard-Negative Mining Dataset Generator for Drift-Sense++ HCR
Generates triplets: (reference_patch, positive_search_patch, hard_negative_search_patch)

The hard negatives are periodic array replicas that FFT-NCC scores highly but are
at wrong physical locations — exactly the failure mode the Siamese re-ranker must solve.

Multi-scale context extraction:
  - 64×64 local patch centered on candidate
  - 128×128 neighborhood context centered on candidate
"""

import os
import sys
import csv
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))

from hf_space.src.pipeline import generate_sample, GenerationParams


# Difficulty presets matching the benchmark generator
DIFFICULTY_CONFIGS = {
    "Easy": GenerationParams(
        dose_search=500.0, shear_amplitude_px=0.5, drift_jitter_px=0.2,
        detector_noise_sigma_search=3.0,
    ),
    "Medium": GenerationParams(
        dose_search=200.0, shear_amplitude_px=1.5, drift_jitter_px=0.5,
        detector_noise_sigma_search=5.0,
    ),
    "Hard": GenerationParams(
        dose_search=100.0, shear_amplitude_px=3.0, drift_jitter_px=1.0,
        detector_noise_sigma_search=8.0,
        speckle_sigma=0.1, charging_streak_prob=1.0, charging_streak_intensity=1.0,
    ),
    "Adversarial": GenerationParams(
        dose_search=55.0, shear_amplitude_px=4.5, drift_jitter_px=2.0,
        detector_noise_sigma_search=12.0,
        speckle_sigma=0.25, salt_pepper_prob=0.008,
        charging_streak_prob=3.0, charging_streak_intensity=2.0,
    ),
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
    return g


def extract_patch(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Extract a square patch centered at (cx, cy) with border reflection padding."""
    h, w = image.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    # Handle boundary with reflection padding
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


def mine_hard_negatives_from_sample(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    gt_x: float,
    gt_y: float,
    top_k: int = 20,
    min_dist: int = 12,
    correct_radius: float = 8.0,
) -> dict:
    """
    Run FFT-NCC on a sample, extract Top-K candidates, classify as
    positive (within correct_radius of GT) or hard-negative (wrong but high score).

    Returns multi-scale patches for each candidate.
    """
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    ref_norm = normalize_intensity(ref_100)
    ref_grad = extract_gradient(ref_100)
    search_norm = normalize_intensity(search_img)
    search_grad = extract_gradient(search_img)

    # Multi-channel FFT retrieval
    c_i = cv2.matchTemplate(search_norm, ref_norm, cv2.TM_CCOEFF_NORMED)
    c_g = cv2.matchTemplate(search_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    c_combo = 0.55 * c_i + 0.45 * c_g

    # Extract Top-K with NMS
    work = c_combo.copy()
    h, w = work.shape
    candidates = []
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + 50.0
        cy = py + 50.0
        err = np.hypot(cx - gt_x, cy - gt_y)
        is_correct = err <= correct_radius

        candidates.append({
            "cx": cx, "cy": cy,
            "ncc_score": float(max_val),
            "error_px": float(err),
            "is_correct": is_correct,
        })

        y1, y2 = max(0, py - min_dist), min(h, py + min_dist + 1)
        x1, x2 = max(0, px - min_dist), min(w, px + min_dist + 1)
        work[y1:y2, x1:x2] = -999.0

    # Extract multi-scale patches for each candidate
    for cand in candidates:
        cx, cy = cand["cx"], cand["cy"]
        cand["patch_64"] = extract_patch(search_img, cx, cy, 64)
        cand["patch_128"] = extract_patch(search_img, cx, cy, 128)

    # Reference patches (centered)
    ref_patch_64 = cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA)
    ref_patch_128 = cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA)

    return {
        "ref_patch_64": ref_patch_64,
        "ref_patch_128": ref_patch_128,
        "candidates": candidates,
        "num_correct": sum(1 for c in candidates if c["is_correct"]),
        "num_wrong": sum(1 for c in candidates if not c["is_correct"]),
    }


def generate_training_dataset(
    output_dir: str,
    num_samples: int = 600,
    seed_base: int = 50000,
):
    """
    Generate a large training dataset with hard-negative mining.

    For each synthetic sample:
    1. Generate reference + search image pair
    2. Run FFT-NCC to get Top-20 candidates
    3. Classify candidates as positive or hard-negative
    4. Save multi-scale patches as numpy arrays

    Dataset structure:
    output_dir/
      triplets/
        XXXX_ref_64.npy, XXXX_ref_128.npy
        XXXX_pos_64.npy, XXXX_pos_128.npy
        XXXX_neg_YY_64.npy, XXXX_neg_YY_128.npy
      manifest.csv
    """
    os.makedirs(os.path.join(output_dir, "triplets"), exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    fieldnames = [
        "triplet_id", "sample_id", "difficulty", "architecture",
        "ref_64_path", "ref_128_path",
        "pos_64_path", "pos_128_path",
        "neg_64_path", "neg_128_path",
        "pos_ncc", "neg_ncc", "pos_error", "neg_error",
        "gt_x", "gt_y",
    ]

    difficulties = list(DIFFICULTY_CONFIGS.keys())
    triplet_count = 0
    stats = {"total_samples": 0, "samples_with_positives": 0, "total_triplets": 0,
             "hard_negatives_mined": 0}

    t0 = time.time()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_idx in range(num_samples):
            seed = seed_base + sample_idx * 31
            rng = np.random.default_rng(seed)
            diff = difficulties[sample_idx % len(difficulties)]
            arch = ARCHITECTURES[sample_idx % len(ARCHITECTURES)]
            params = DIFFICULTY_CONFIGS[diff]

            try:
                sample = generate_sample(arch, rng, params)
            except Exception as e:
                print(f"  [WARN] Sample {sample_idx} generation failed: {e}")
                continue

            ref_img = sample["reference_img"]
            search_img = sample["search_img"]
            gt_x = float(sample["gt_x"])
            gt_y = float(sample["gt_y"])

            stats["total_samples"] += 1

            # Mine hard negatives
            mined = mine_hard_negatives_from_sample(
                ref_img, search_img, gt_x, gt_y, top_k=20
            )

            if mined["num_correct"] == 0:
                # No correct candidate found in top-20 — still useful as all-negative sample
                continue

            stats["samples_with_positives"] += 1

            # Get best positive
            positives = [c for c in mined["candidates"] if c["is_correct"]]
            best_pos = min(positives, key=lambda c: c["error_px"])

            # Get hard negatives (wrong but high NCC score)
            negatives = [c for c in mined["candidates"] if not c["is_correct"]]
            negatives.sort(key=lambda c: c["ncc_score"], reverse=True)

            # Take up to 5 hardest negatives per sample
            hard_negs = negatives[:5]

            if len(hard_negs) == 0:
                continue

            stats["hard_negatives_mined"] += len(hard_negs)

            for neg in hard_negs:
                triplet_dir = os.path.join(output_dir, "triplets")

                ref_64_path = os.path.join(triplet_dir, f"{triplet_count:05d}_ref_64.npy")
                ref_128_path = os.path.join(triplet_dir, f"{triplet_count:05d}_ref_128.npy")
                pos_64_path = os.path.join(triplet_dir, f"{triplet_count:05d}_pos_64.npy")
                pos_128_path = os.path.join(triplet_dir, f"{triplet_count:05d}_pos_128.npy")
                neg_64_path = os.path.join(triplet_dir, f"{triplet_count:05d}_neg_64.npy")
                neg_128_path = os.path.join(triplet_dir, f"{triplet_count:05d}_neg_128.npy")

                np.save(ref_64_path, mined["ref_patch_64"])
                np.save(ref_128_path, mined["ref_patch_128"])
                np.save(pos_64_path, best_pos["patch_64"])
                np.save(pos_128_path, best_pos["patch_128"])
                np.save(neg_64_path, neg["patch_64"])
                np.save(neg_128_path, neg["patch_128"])

                writer.writerow({
                    "triplet_id": triplet_count,
                    "sample_id": sample_idx,
                    "difficulty": diff,
                    "architecture": arch,
                    "ref_64_path": ref_64_path,
                    "ref_128_path": ref_128_path,
                    "pos_64_path": pos_64_path,
                    "pos_128_path": pos_128_path,
                    "neg_64_path": neg_64_path,
                    "neg_128_path": neg_128_path,
                    "pos_ncc": round(best_pos["ncc_score"], 4),
                    "neg_ncc": round(neg["ncc_score"], 4),
                    "pos_error": round(best_pos["error_px"], 2),
                    "neg_error": round(neg["error_px"], 2),
                    "gt_x": round(gt_x, 2),
                    "gt_y": round(gt_y, 2),
                })

                triplet_count += 1

            if (sample_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (sample_idx + 1) / elapsed
                print(f"  [{sample_idx+1}/{num_samples}] {triplet_count} triplets mined | "
                      f"{rate:.1f} samples/s | elapsed {elapsed:.0f}s")

    stats["total_triplets"] = triplet_count
    elapsed_total = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  HARD-NEGATIVE MINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total synthetic samples generated: {stats['total_samples']}")
    print(f"  Samples with correct candidate in Top-20: {stats['samples_with_positives']}")
    print(f"  Total triplets mined: {stats['total_triplets']}")
    print(f"  Hard negatives collected: {stats['hard_negatives_mined']}")
    print(f"  Time elapsed: {elapsed_total:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")

    return stats


def generate_held_out_test_set(
    output_dir: str,
    num_samples: int = 200,
    seed_base: int = 999000,
):
    """
    Generate a completely unseen held-out test set with different seeds.
    No hard-negative mining — just plain image pairs with ground truth.
    """
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "search"), exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    fieldnames = [
        "id", "difficulty", "architecture", "reference_path", "search_path",
        "gt_x", "gt_y", "seed"
    ]

    difficulties = list(DIFFICULTY_CONFIGS.keys())

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_samples):
            seed = seed_base + i * 43
            rng = np.random.default_rng(seed)
            diff = difficulties[i % len(difficulties)]
            arch = ARCHITECTURES[i % len(ARCHITECTURES)]
            params = DIFFICULTY_CONFIGS[diff]

            try:
                sample = generate_sample(arch, rng, params)
            except Exception as e:
                print(f"  [WARN] Test sample {i} failed: {e}")
                continue

            ref_path = os.path.join(output_dir, "reference", f"{i:04d}.png")
            search_path = os.path.join(output_dir, "search", f"{i:04d}.png")

            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            writer.writerow({
                "id": i,
                "difficulty": diff,
                "architecture": arch,
                "reference_path": ref_path,
                "search_path": search_path,
                "gt_x": round(float(sample["gt_x"]), 2),
                "gt_y": round(float(sample["gt_y"]), 2),
                "seed": seed,
            })

    print(f"Generated {num_samples}-case held-out test set in '{output_dir}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hard-Negative Mining Dataset Generator")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--test-samples", type=int, default=200)
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        print("=" * 70)
        print("  PHASE 1: Generating training triplets with hard-negative mining...")
        print("=" * 70)
        generate_training_dataset(
            output_dir="data/hcr_train",
            num_samples=args.train_samples,
            seed_base=50000,
        )

    if args.mode in ("test", "both"):
        print("\n" + "=" * 70)
        print("  PHASE 2: Generating held-out test set (unseen seeds)...")
        print("=" * 70)
        generate_held_out_test_set(
            output_dir="data/hcr_test",
            num_samples=args.test_samples,
            seed_base=999000,
        )
