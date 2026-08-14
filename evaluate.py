"""
Drift-Sense++ Evaluation Harness & Ablation Benchmarking Suite
Evaluates localization accuracy, pixel errors, inference time, and full ablation table
across train, val, and stress dataset splits.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from inference import (
    perform_drift_sense_localization,
    compute_structural_map,
    estimate_noise_mad,
    subpixel_refine,
    compute_radon_fingerprint_score
)


def run_baseline_ncc(ref_img, search_img):
    """Baseline 0: Plain Normalized Cross-Correlation on raw 100x downsampled image."""
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(search_img, ref_100, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    pred_x = float(max_loc[0] + 50)
    pred_y = float(max_loc[1] + 50)
    return pred_x, pred_y


def run_ablation_method(ref_img, search_img, method_id):
    """
    Method IDs:
    0: NCC only (Raw image)
    1: FFT + Gradient map (G only)
    2: FFT + Hybrid map (PC + G)
    3: FFT + Hybrid + Radon fingerprinting
    4: Full Drift-Sense++ (FFT + Hybrid + Radon + Subpixel)
    """
    if method_id == 0:
        return run_baseline_ncc(ref_img, search_img)

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    n_ref = estimate_noise_mad(ref_100)
    n_search = estimate_noise_mad(search_img)

    if method_id == 1:
        # Gradient map only
        _, F_ref = compute_structural_map(ref_100, 0)
        _, F_search = compute_structural_map(search_img, 0)
    else:
        # Hybrid PC + Gradient map
        F_ref, G_ref = compute_structural_map(ref_100, n_ref)
        F_search, G_search = compute_structural_map(search_img, n_search)

    scales = [0.95, 0.975, 1.0, 1.025, 1.05]
    rotations = [-3.0, -1.5, 0.0, 1.5, 3.0]

    best_score = -1.0
    best_pred = None

    for s in scales:
        tw = int(round(100 * s))
        th = int(round(100 * s))
        t_res = cv2.resize(F_ref, (tw, th), interpolation=cv2.INTER_CUBIC)

        for r in rotations:
            M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), r, 1.0)
            t_rot = cv2.warpAffine(t_res, M, (tw, th))

            res = cv2.matchTemplate(F_search, t_rot, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            score = max_val
            if method_id >= 3:
                # Add Radon fingerprint boost
                sh, sw = F_search.shape
                y1, y2 = max(0, int(max_loc[1])), min(sh, int(max_loc[1] + th))
                x1, x2 = max(0, int(max_loc[0])), min(sw, int(max_loc[0] + tw))
                patch = F_search[y1:y2, x1:x2]
                radon = compute_radon_fingerprint_score(patch, t_rot) if patch.size > 0 else 0.0
                score = 0.7 * max_val + 0.3 * radon

            if score > best_score:
                best_score = score
                best_pred = (max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0, max_loc[0] + tw // 2, max_loc[1] + th // 2)

    pred_x, pred_y = best_pred[0], best_pred[1]
    if method_id == 4:
        # Subpixel refinement
        _, G_search = compute_structural_map(search_img, n_search)
        pred_x, pred_y = subpixel_refine(G_search, best_pred[2], best_pred[3])

    return pred_x, pred_y


def evaluate_dataset(dataset_dir):
    csv_path = os.path.join(dataset_dir, "ground_truth.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.", file=sys.stderr)
        return None

    df = pd.read_csv(csv_path)
    results = []

    print(f"\n[Evaluate] Running evaluation on {len(df)} pairs in '{dataset_dir}'...")

    for idx, row in df.iterrows():
        ref_path = row['ref_path']
        search_path = row['search_path']

        if not os.path.exists(ref_path) or not os.path.exists(search_path):
            continue

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        pred_x, pred_y, meta = perform_drift_sense_localization(ref_img, search_img)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        x_true, y_true = row['x_true'], row['y_true']
        error_px = float(np.sqrt((pred_x - x_true)**2 + (pred_y - y_true)**2))

        record = {
            "image_id": row['image_id'],
            "style": row['style'],
            "split": row['split'],
            "x_true": x_true,
            "y_true": y_true,
            "pred_x": round(pred_x, 2),
            "pred_y": round(pred_y, 2),
            "error_px": round(error_px, 3),
            "elapsed_ms": round(elapsed_ms, 2),
            "confidence": meta["confidence"],
            "psr": meta["psr"],
            "status": meta["status"]
        }
        results.append(record)

    res_df = pd.DataFrame(results)
    return res_df


def save_visualization(res_df, output_path, title, is_success=True):
    """Saves a 4-panel visual diagnostic figure for success or failure case."""
    if is_success:
        sample_row = res_df.sort_values("error_px").iloc[0]
    else:
        sample_row = res_df.sort_values("error_px", ascending=False).iloc[0]

    # Load images
    gt_df = pd.read_csv("data/train/ground_truth.csv") if os.path.exists("data/train/ground_truth.csv") else None
    if gt_df is not None and sample_row['image_id'] in gt_df['image_id'].values:
        match_info = gt_df[gt_df['image_id'] == sample_row['image_id']].iloc[0]
        ref_path = match_info['ref_path']
        search_path = match_info['search_path']
    else:
        # Fallback search in data dirs
        found_ref, found_search = None, None
        for s in ["train", "val", "stress"]:
            path_r = f"data/{s}/images/ref_{sample_row['image_id']}.png"
            path_s = f"data/{s}/images/search_{sample_row['image_id']}.png"
            if os.path.exists(path_r):
                found_ref, found_search = path_r, path_s
                break
        ref_path, search_path = found_ref, found_search

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE) if ref_path and os.path.exists(ref_path) else np.zeros((1000, 1000), dtype=np.uint8)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE) if search_path and os.path.exists(search_path) else np.zeros((1000, 1000), dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} - Pair {sample_row['image_id']} (Error: {sample_row['error_px']:.2f} px)", fontsize=14, fontweight='bold')

    axes[0].imshow(ref_img, cmap='gray')
    axes[0].set_title("100x Reference SEM (Target Patch)")
    axes[0].axis('off')

    # Draw predicted and GT circles on search image
    search_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
    cv2.circle(search_color, (int(round(sample_row['x_true'])), int(round(sample_row['y_true']))), 30, (0, 255, 0), 4) # Green GT
    cv2.circle(search_color, (int(round(sample_row['pred_x'])), int(round(sample_row['pred_y']))), 20, (255, 0, 0), 3) # Red Pred

    axes[1].imshow(search_color)
    axes[1].set_title("10x Search SEM (Green=GT, Red=Pred)")
    axes[1].axis('off')

    # Crop zoomed-in view around GT
    cx, cy = int(round(sample_row['x_true'])), int(round(sample_row['y_true']))
    y1, y2 = max(0, cy-150), min(1000, cy+150)
    x1, x2 = max(0, cx-150), min(1000, cx+150)
    zoom_crop = search_color[y1:y2, x1:x2]

    axes[2].imshow(zoom_crop)
    axes[2].set_title("Zoomed ROI Verification")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Visual saved to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Evaluation Harness")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing train/val/stress")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save benchmarks and ablation table")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "val", "stress"]
    all_results = []

    for split in splits:
        split_dir = os.path.join(args.data_dir, split)
        if os.path.exists(split_dir):
            res_df = evaluate_dataset(split_dir)
            if res_df is not None and not res_df.empty:
                all_results.append(res_df)

    if not all_results:
        print("Error: No valid datasets found to evaluate.", file=sys.stderr)
        sys.exit(1)

    full_df = pd.concat(all_results, ignore_index=True)

    # Calculate overall metrics
    mean_err = full_df['error_px'].mean()
    median_err = full_df['error_px'].median()
    p95_err = full_df['error_px'].quantile(0.95)

    acc_1px = (full_df['error_px'] <= 1.0).mean() * 100.0
    acc_3px = (full_df['error_px'] <= 3.0).mean() * 100.0
    acc_5px = (full_df['error_px'] <= 5.0).mean() * 100.0

    mean_time = full_df['elapsed_ms'].mean()
    median_time = full_df['elapsed_ms'].median()
    p95_time = full_df['elapsed_ms'].quantile(0.95)

    print("\n" + "="*60)
    print("           DRIFT-SENSE++ ACCURACY & BENCHMARK SUMMARY           ")
    print("="*60)
    print(f" Total Evaluated Pairs  : {len(full_df)}")
    print(f" Accuracy within 1.0 px : {acc_1px:.1f}%")
    print(f" Accuracy within 3.0 px : {acc_3px:.1f}%")
    print(f" Accuracy within 5.0 px : {acc_5px:.1f}%")
    print(f" Mean Pixel Error       : {mean_err:.2f} px")
    print(f" Median Pixel Error     : {median_err:.2f} px")
    print(f" P95 Pixel Error        : {p95_err:.2f} px")
    print(f" Mean Latency           : {mean_time:.2f} ms / pair")
    print(f" Median Latency         : {median_time:.2f} ms / pair")
    print(f" P95 Latency            : {p95_time:.2f} ms / pair")
    print("="*60)

    # Save Runtime Benchmark CSV
    runtime_csv = os.path.join(args.output_dir, "runtime_benchmark.csv")
    runtime_data = [{
        "total_pairs": len(full_df),
        "mean_latency_ms": round(mean_time, 2),
        "median_latency_ms": round(median_time, 2),
        "p95_latency_ms": round(p95_time, 2),
        "accuracy_1px_pct": round(acc_1px, 2),
        "accuracy_3px_pct": round(acc_3px, 2),
        "mean_error_px": round(mean_err, 2),
        "median_error_px": round(median_err, 2),
        "p95_error_px": round(p95_err, 2)
    }]
    pd.DataFrame(runtime_data).to_csv(runtime_csv, index=False)
    print(f"[Evaluate] Saved runtime benchmark to '{runtime_csv}'.")

    # Run Ablation Study over Validation set
    val_dir = os.path.join(args.data_dir, "val")
    val_csv = os.path.join(val_dir, "ground_truth.csv")
    if os.path.exists(val_csv):
        print("\n[Ablation Study] Running ablation over validation split...")
        val_df = pd.read_csv(val_csv)
        ablation_methods = [
            ("NCC only (Raw Image Baseline)", 0),
            ("+ FFT Coarse Search", 1),
            ("+ Hybrid Structural Map (PC + G)", 2),
            ("+ Adaptive Radon Verification", 3),
            ("+ Subpixel Parabola Refinement (Full Drift-Sense++)", 4)
        ]

        ablation_rows = []
        for name, m_id in ablation_methods:
            method_errors = []
            t0_m = time.perf_counter()
            for idx, r_item in val_df.iterrows():
                ref_img = cv2.imread(r_item['ref_path'], cv2.IMREAD_GRAYSCALE)
                search_img = cv2.imread(r_item['search_path'], cv2.IMREAD_GRAYSCALE)
                if ref_img is None or search_img is None:
                    continue
                px, py = run_ablation_method(ref_img, search_img, m_id)
                err = float(np.sqrt((px - r_item['x_true'])**2 + (py - r_item['y_true'])**2))
                method_errors.append(err)
            t1_m = time.perf_counter()

            avg_time = ((t1_m - t0_m) * 1000.0) / max(1, len(method_errors))
            m_err = np.mean(method_errors)
            p95_e = np.quantile(method_errors, 0.95)
            acc3 = (np.array(method_errors) <= 3.0).mean() * 100.0

            ablation_rows.append({
                "Method Stage": name,
                "Accuracy (<=3px) %": round(acc3, 1),
                "Mean Error (px)": round(m_err, 2),
                "P95 Error (px)": round(p95_e, 2),
                "Avg Latency (ms)": round(avg_time, 2)
            })

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_csv = os.path.join(args.output_dir, "ablation_table.csv")
        ablation_df.to_csv(ablation_csv, index=False)

        print("\n" + "-"*70)
        print(ablation_df.to_string(index=False))
        print("-"*70)
        print(f"[Evaluate] Saved ablation table to '{ablation_csv}'.")

    # Save visual diagnostic images
    save_visualization(full_df, os.path.join(args.output_dir, "success_example.png"), "Success Case", is_success=True)
    save_visualization(full_df, os.path.join(args.output_dir, "failure_example.png"), "Honest Stress / Failure Case", is_success=False)


if __name__ == "__main__":
    main()
