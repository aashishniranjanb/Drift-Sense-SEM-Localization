"""
Diagnostics Engine for Drift-Sense Localization
1. Clean-Case Oracle Validation (Verification of ground-truth coordinate correctness on noise/drift/distortion-free pairs)
2. Top-K Candidate Spatial Recall (Top-1, Top-3, Top-5, Top-10) to distinguish retrieval failure from ranking failure
3. 2D Correlation Heatmap & Ambiguity Visualization
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))
from hf_space.src.pipeline import GenerationParams, generate_sample


def extract_distinct_peaks(corr_map: np.ndarray, top_k: int = 10, min_distance: int = 15) -> list[dict]:
    """
    Extracts the top-K distinct spatial local maxima from a 2D correlation map
    using peak non-maximum suppression (NMS) with minimum spatial distance.
    """
    corr_work = corr_map.copy()
    h, w = corr_work.shape
    peaks = []

    for _ in range(top_k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr_work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        peaks.append({
            "x": int(px),
            "y": int(py),
            "score": float(max_val)
        })
        # Suppress local neighborhood around this peak
        y1 = max(0, py - min_distance)
        y2 = min(h, py + min_distance + 1)
        x1 = max(0, px - min_distance)
        x2 = min(w, px + min_distance + 1)
        corr_work[y1:y2, x1:x2] = -999.0

    return peaks


def run_clean_oracle_validation(num_samples: int = 20) -> bool:
    """
    Synthesizes perfectly clean image pairs (noise=0, blur=0, drift=0, shear=0, scale=1.0)
    and tests whether plain 100x100 template FFT-NCC retrieves the exact ground truth (<= 1.0 px).
    """
    print("=" * 70)
    print("        LEVEL 1: CLEAN-CASE ORACLE GROUND-TRUTH VALIDATION        ")
    print("=" * 70)

    clean_params = GenerationParams(
        beam_spot_size_nm=0.0,
        dose_reference=50000.0,
        dose_search=50000.0,
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=0.0,
        detector_noise_sigma_search=0.0,
        astigmatism_ratio=1.0,
        vignette_strength=0.0,
        gamma=1.0,
        barrel_distortion_k=0.0,
        charging_streak_prob=0.0,
        charging_streak_intensity=0.0,
        speckle_sigma=0.0,
        salt_pepper_prob=0.0
    )

    architectures = ["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]
    errors = []

    for i in range(num_samples):
        arch = architectures[i % len(architectures)]
        rng = np.random.default_rng(1000 + i * 37)
        sample = generate_sample(arch, rng, clean_params)

        ref_img = sample["reference_img"]
        search_img = sample["search_img"]
        gt_x, gt_y = sample["gt_x"], sample["gt_y"]

        # Downsample reference to 100x100
        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

        # Plain FFT Cross Correlation (TM_CCOEFF_NORMED)
        res = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        pred_x = max_loc[0] + 50.0
        pred_y = max_loc[1] + 50.0

        err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        errors.append(err)

        status = "[PASS]" if err <= 1.5 else "[FAIL]"
        print(f"Sample {i+1:02d} [{arch:<12s}]: Pred=({pred_x:.1f}, {pred_y:.1f}) | GT=({gt_x:.1f}, {gt_y:.1f}) | Err={err:.3f} px | Score={max_val:.4f} {status}")

    errors_arr = np.array(errors)
    pass_rate = float(np.mean(errors_arr <= 1.5) * 100)
    print("-" * 70)
    print(f"Clean Oracle Validation Result: {pass_rate:.1f}% within 1.5 px (Mean error: {np.mean(errors_arr):.3f} px)")
    print("=" * 70)
    return pass_rate >= 95.0


def run_topk_recall_diagnostic(manifest_path: str = "data/benchmark_120/manifest.csv") -> pd.DataFrame:
    """
    Evaluates Top-1, Top-3, Top-5, Top-10 candidate spatial recall across all 120 benchmark cases
    for:
    - Intensity FFT-NCC
    - Gradient FFT-NCC
    - Multi-Scale Dual Channel FFT
    """
    print("\n" + "=" * 70)
    print("        LEVEL 2: TOP-K SPATIAL CANDIDATE RETRIEVAL RECALL DIAGNOSTIC        ")
    print("=" * 70)

    if not os.path.exists(manifest_path):
        print(f"Error: manifest '{manifest_path}' not found!")
        return None

    df_manifest = pd.read_csv(manifest_path)

    methods = ["Intensity_FFT", "Gradient_FFT", "MultiScale_DualChannel"]
    recalls = {m: {"Top-1": 0, "Top-3": 0, "Top-5": 0, "Top-10": 0, "total": 0} for m in methods}
    diff_recalls = {}

    for idx, row in df_manifest.iterrows():
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        diff = row["difficulty"]

        if diff not in diff_recalls:
            diff_recalls[diff] = {m: {"Top-1": 0, "Top-10": 0, "total": 0} for m in methods}

        # 1. Intensity FFT
        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        corr_int = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        peaks_int = extract_distinct_peaks(corr_int, top_k=10, min_distance=15)
        cands_int = [(p["x"] + 50.0, p["y"] + 50.0) for p in peaks_int]

        # 2. Gradient FFT
        ref_g = cv2.magnitude(cv2.Scharr(ref_100.astype(np.float32)/255.0, cv2.CV_32F, 1, 0),
                              cv2.Scharr(ref_100.astype(np.float32)/255.0, cv2.CV_32F, 0, 1))
        search_g = cv2.magnitude(cv2.Scharr(search_img.astype(np.float32)/255.0, cv2.CV_32F, 1, 0),
                                cv2.Scharr(search_img.astype(np.float32)/255.0, cv2.CV_32F, 0, 1))
        corr_grad = cv2.matchTemplate(search_g, ref_g, cv2.TM_CCOEFF_NORMED)
        peaks_grad = extract_distinct_peaks(corr_grad, top_k=10, min_distance=15)
        cands_grad = [(p["x"] + 50.0, p["y"] + 50.0) for p in peaks_grad]

        # 3. Multi-Scale Dual Channel (0.95 - 1.05)
        all_cands_dual = []
        for s in [0.95, 0.98, 1.00, 1.02, 1.05]:
            tw = int(round(100 * s))
            th = int(round(100 * s))
            r_s = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
            r_sg = cv2.magnitude(cv2.Scharr(r_s.astype(np.float32)/255.0, cv2.CV_32F, 1, 0),
                                 cv2.Scharr(r_s.astype(np.float32)/255.0, cv2.CV_32F, 0, 1))

            c_i = cv2.matchTemplate(search_img.astype(np.float32), r_s.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            c_g = cv2.matchTemplate(search_g, r_sg, cv2.TM_CCOEFF_NORMED)
            c_combo = 0.5 * c_i + 0.5 * c_g

            p_combo = extract_distinct_peaks(c_combo, top_k=6, min_distance=15)
            for p in p_combo:
                all_cands_dual.append({
                    "x": p["x"] + tw / 2.0,
                    "y": p["y"] + th / 2.0,
                    "score": p["score"]
                })

        all_cands_dual.sort(key=lambda c: c["score"], reverse=True)
        # NMS on multi-scale candidates
        unique_dual = []
        for c in all_cands_dual:
            if not any(np.hypot(c["x"] - u["x"], c["y"] - u["y"]) < 12 for u in unique_dual):
                unique_dual.append(c)
            if len(unique_dual) >= 10:
                break
        cands_dual = [(c["x"], c["y"]) for c in unique_dual]

        method_cands = {
            "Intensity_FFT": cands_int,
            "Gradient_FFT": cands_grad,
            "MultiScale_DualChannel": cands_dual
        }

        for m_name, c_list in method_cands.items():
            recalls[m_name]["total"] += 1
            diff_recalls[diff][m_name]["total"] += 1

            for k, k_name in [(1, "Top-1"), (3, "Top-3"), (5, "Top-5"), (10, "Top-10")]:
                sub_list = c_list[:k]
                has_match = any(np.hypot(cx - gt_x, cy - gt_y) <= 5.0 for cx, cy in sub_list)
                if has_match:
                    recalls[m_name][k_name] += 1
                    if k in (1, 10):
                        diff_recalls[diff][m_name][k_name] += 1

    summary_rows = []
    for m_name in methods:
        tot = recalls[m_name]["total"]
        t1 = (recalls[m_name]["Top-1"] / tot) * 100
        t3 = (recalls[m_name]["Top-3"] / tot) * 100
        t5 = (recalls[m_name]["Top-5"] / tot) * 100
        t10 = (recalls[m_name]["Top-10"] / tot) * 100
        summary_rows.append({
            "Retrieval Method": m_name,
            "Top-1 Recall (%)": round(t1, 2),
            "Top-3 Recall (%)": round(t3, 2),
            "Top-5 Recall (%)": round(t5, 2),
            "Top-10 Recall (%)": round(t10, 2),
            "Retrieval vs Ranking Gap": round(t10 - t1, 2)
        })

    df_topk = pd.DataFrame(summary_rows)
    print(df_topk.to_string(index=False))
    print("-" * 70)

    print("\nTop-10 Recall Breakdown by Difficulty:")
    for diff in ["Easy", "Medium", "Hard", "Adversarial"]:
        print(f"  [{diff:<12s}]: ", end="")
        for m_name in methods:
            t = diff_recalls[diff][m_name]["total"]
            r10 = (diff_recalls[diff][m_name]["Top-10"] / t) * 100 if t > 0 else 0
            print(f"{m_name} Top-10 = {r10:.1f}% | ", end="")
        print()
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    df_topk.to_csv("results/topk_recall_diagnostic.csv", index=False)
    return df_topk


def generate_correlation_visualizations(sample_indices: list[int] = [0, 30, 60, 90], out_dir: str = "results/diagnostics"):
    """
    Generates side-by-side diagnostic visualizations showing Reference, Search with Ground Truth,
    and 2D Correlation Heatmap with Top-10 Candidate Peaks.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = "data/benchmark_120/manifest.csv"
    if not os.path.exists(manifest_path):
        return

    df_manifest = pd.read_csv(manifest_path)

    for idx in sample_indices:
        if idx >= len(df_manifest):
            continue
        row = df_manifest.iloc[idx]
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        diff = row["difficulty"]
        arch = row["architecture"]

        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        corr_map = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        peaks = extract_distinct_peaks(corr_map, top_k=10, min_distance=15)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Reference
        axs[0].imshow(ref_img, cmap="gray")
        axs[0].set_title(f"Reference Image (100x)\n{arch}")
        axs[0].axis("off")

        # 2. Search Image with GT and Top Candidate
        search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
        # GT Box (100x100)
        gx1, gy1 = int(round(gt_x - 50)), int(round(gt_y - 50))
        gx2, gy2 = int(round(gt_x + 50)), int(round(gt_y + 50))
        cv2.rectangle(search_rgb, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3) # Green for GT
        cv2.circle(search_rgb, (int(round(gt_x)), int(round(gt_y))), 6, (0, 255, 0), -1)

        # Top candidate
        if len(peaks) > 0:
            c1_x = peaks[0]["x"] + 50
            c1_y = peaks[0]["y"] + 50
            cv2.circle(search_rgb, (c1_x, c1_y), 6, (255, 0, 0), -1) # Red for Top-1
            cv2.rectangle(search_rgb, (c1_x - 50, c1_y - 50), (c1_x + 50, c1_y + 50), (255, 0, 0), 2)

        axs[1].imshow(search_rgb)
        axs[1].set_title(f"Search Image (10x) - [{diff}]\nGreen=GT, Red=Top-1 Pred")
        axs[1].axis("off")

        # 3. Correlation Map with all Top-10 peaks
        im_corr = axs[2].imshow(corr_map, cmap="inferno")
        # Overlay peaks
        for i, p in enumerate(peaks):
            color = "cyan" if i == 0 else "white"
            axs[2].plot(p["x"], p["y"], marker="o", color=color, markersize=5)
            axs[2].text(p["x"] + 10, p["y"] + 10, f"#{i+1}", color=color, fontsize=8)

        # Overlay GT peak location in correlation map (shift by -50)
        gt_corr_x = np.clip(gt_x - 50, 0, corr_map.shape[1] - 1)
        gt_corr_y = np.clip(gt_y - 50, 0, corr_map.shape[0] - 1)
        axs[2].plot(gt_corr_x, gt_corr_y, marker="x", color="lime", markersize=10, mew=2, label="True GT Peak")

        axs[2].set_title(f"2D Correlation Heatmap\nTop-10 Candidates (Lime 'X' = GT)")
        axs[2].axis("off")
        plt.colorbar(im_corr, ax=axs[2], fraction=0.046, pad=0.04)

        save_path = os.path.join(out_dir, f"diag_sample_{idx:03d}_{diff}_{arch}.png")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved diagnostic correlation map: '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for Drift-Sense Localization")
    parser.add_argument("--mode", type=str, default="all", choices=["oracle", "topk", "viz", "all"])
    args = parser.parse_args()

    if args.mode in ("oracle", "all"):
        run_clean_oracle_validation(num_samples=20)

    if args.mode in ("topk", "all"):
        run_topk_recall_diagnostic()

    if args.mode in ("viz", "all"):
        generate_correlation_visualizations()


if __name__ == "__main__":
    main()
