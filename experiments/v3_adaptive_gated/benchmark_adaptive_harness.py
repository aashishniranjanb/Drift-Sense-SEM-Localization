"""
Benchmark & Profiling Harness for Drift-Sense++ Adaptive Structural Registration
Compares:
- V0: Baseline ZNCC
- V1: Plain FFT-NCC
- V4: Fixed Multi-Scale 25-Grid
- Drift-Sense++ Adaptive Pipeline
Profiles Path Distribution (Fast, Normal, Hard), Latency P95, and Accuracy Pareto Frontier.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import pandas as pd

from inference_adaptive import perform_adaptive_localization, extract_gradient_and_orientation, normalize_intensity
from inference_v5 import perform_drift_sense_v5
from hf_space.baseline_solution.zncc import zncc_match

MANIFEST_PATH = "data/benchmark_120/manifest.csv"


def run_fft_ncc(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, float]:
    ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search_img.astype(np.float32), ref_down.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr)
    return float(max_loc[0] + 50), float(max_loc[1] + 50), float(max_val)


def main():
    print("=" * 95)
    print("      DRIFT-SENSE++ ADAPTIVE PIPELINE BENCHMARK & PARETO PROFILING (120 SAMPLES)      ")
    print("=" * 95)

    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest '{MANIFEST_PATH}' not found!")
        return

    df_manifest = pd.read_csv(MANIFEST_PATH)

    variants = [
        "V0_ZNCC",
        "V1_FFT_NCC",
        "V4_Fixed_MultiScale_25Grid",
        "DriftSense_PlusPlus_Adaptive"
    ]

    results = []
    diff_breakdown = {}
    path_counts = {"FAST_PATH": 0, "NORMAL_PATH": 0, "HARD_PATH": 0}
    path_accuracies = {"FAST_PATH": [], "NORMAL_PATH": [], "HARD_PATH": []}

    for v_id in variants:
        print(f"\n--> Running Evaluation for: [{v_id}]...")
        latencies = []
        errors = []
        acc_1px, acc_3px, acc_5px, acc_10px = [], [], [], []
        diff_accs = {d: [] for d in ["Easy", "Medium", "Hard", "Adversarial"]}

        for idx, row in df_manifest.iterrows():
            ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            diff = row["difficulty"]

            t0 = time.perf_counter()

            if v_id == "V0_ZNCC":
                res = zncc_match(ref_img, search_img)
                x_pred, y_pred = float(res["x"]), float(res["y"])
            elif v_id == "V1_FFT_NCC":
                x_pred, y_pred, _ = run_fft_ncc(ref_img, search_img)
            elif v_id == "V4_Fixed_MultiScale_25Grid":
                x_pred, y_pred, meta = perform_drift_sense_v5(ref_img, search_img, scales=(0.95, 0.98, 1.00, 1.02, 1.05), rotations=(-3.0, -1.5, 0.0, 1.5, 3.0), use_multi_anchor=False)
            elif v_id == "DriftSense_PlusPlus_Adaptive":
                x_pred, y_pred, meta = perform_adaptive_localization(ref_img, search_img, verbose=False)
                p_name = meta.get("path_taken", "NORMAL_PATH")
                path_counts[p_name] += 1
                err_tmp = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
                path_accuracies[p_name].append(err_tmp <= 5.0)

            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
            errors.append(err)

            acc_1px.append(err <= 1.0)
            acc_3px.append(err <= 3.0)
            acc_5px.append(err <= 5.0)
            acc_10px.append(err <= 10.0)
            diff_accs[diff].append(err <= 5.0)

        err_arr = np.array(errors)
        lat_arr = np.array(latencies)

        diff_breakdown[v_id] = {d: round(float(np.mean(diff_accs[d])) * 100, 1) for d in diff_accs}

        results.append({
            "Variant": v_id,
            "Acc (<=1px) %": round(float(np.mean(acc_1px)) * 100, 2),
            "Acc (<=3px) %": round(float(np.mean(acc_3px)) * 100, 2),
            "Acc (<=5px) %": round(float(np.mean(acc_5px)) * 100, 2),
            "Acc (<=10px) %": round(float(np.mean(acc_10px)) * 100, 2),
            "Mean Err (px)": round(float(np.mean(err_arr)), 2),
            "Median Err (px)": round(float(np.median(err_arr)), 2),
            "P95 Err (px)": round(float(np.percentile(err_arr, 95)), 2),
            "Mean Lat (ms)": round(float(np.mean(lat_arr)), 2),
            "P95 Lat (ms)": round(float(np.percentile(lat_arr, 95)), 2)
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print("                     DRIFT-SENSE++ BENCHMARK COMPARISON                     ")
    print("=" * 95)
    print(df_res.to_string(index=False))
    print("=" * 95)

    print("\nAccuracy (<=5px) Breakdown by Difficulty Level (%):")
    df_diff = pd.DataFrame(diff_breakdown).T
    print(df_diff.to_string())
    print("=" * 95)

    total_samples = len(df_manifest)
    print("\nDrift-Sense++ Adaptive Execution Path Routing Distribution:")
    for p, count in path_counts.items():
        pct = (count / total_samples) * 100
        p_acc = np.mean(path_accuracies[p]) * 100 if len(path_accuracies[p]) > 0 else 0.0
        print(f"  - [{p:<12s}]: {count:3d} samples ({pct:5.1f}%) | Path Accuracy (<=5px): {p_acc:.1f}%")
    print("=" * 95)

    os.makedirs("results", exist_ok=True)
    out_csv = "results/adaptive_benchmark_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nWrote full benchmark summary to '{out_csv}'")


if __name__ == "__main__":
    main()
