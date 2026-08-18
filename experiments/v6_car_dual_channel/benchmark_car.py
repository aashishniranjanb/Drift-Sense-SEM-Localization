"""
Benchmark Harness for Drift-Sense++ CAR Architecture
Evaluates on the frozen 200-case held-out test set (data/hcr_test).
Reports:
  - Accuracy (<=1px, <=3px, <=5px, <=10px)
  - Mean Error, Median Error, P95 Error, Mean Latency
  - PACE Activation Rate & Correct Override Audit Rate
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd

from inference_car import perform_car_localization
from hf_space.baseline_solution.zncc import zncc_match

ORIGINAL_MANIFEST = "data/benchmark_120/manifest.csv"
HOLDOUT_MANIFEST = "data/hcr_test/manifest.csv"


def run_fft_ncc(ref_img, search_img):
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr)
    return float(max_loc[0] + 50), float(max_loc[1] + 50)


def evaluate_car_benchmark(manifest_path, label=""):
    if not os.path.exists(manifest_path):
        print(f"[SKIP] Manifest not found: {manifest_path}")
        return None

    df = pd.read_csv(manifest_path)
    n = len(df)
    print(f"\n{'='*90}")
    print(f"  Evaluating on: {label} ({n} samples)")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*90}")

    methods = {
        "ZNCC_Baseline": [],
        "FFT_NCC": [],
        "CAR_Pipeline": [],
    }

    override_audit = {
        "pace_activations": 0,
        "fft_overrides": 0,
        "correct_overrides": 0,
        "harmful_overrides": 0,
    }

    for method_name in methods:
        print(f"\n  --> Method: [{method_name}]")
        errors = []
        latencies = []
        per_diff = {}

        for idx, row in df.iterrows():
            ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            diff = row.get("difficulty", "Unknown")

            t0 = time.perf_counter()

            if method_name == "ZNCC_Baseline":
                res = zncc_match(ref_img, search_img)
                x_pred, y_pred = float(res["x"]), float(res["y"])
            elif method_name == "FFT_NCC":
                x_pred, y_pred = run_fft_ncc(ref_img, search_img)
            elif method_name == "CAR_Pipeline":
                x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)

                # Audit PACE activations and overrides
                fft_x, fft_y = run_fft_ncc(ref_img, search_img)
                fft_err = float(np.hypot(fft_x - gt_x, fft_y - gt_y))
                car_err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))

                if meta.get("pace_activated", False):
                    override_audit["pace_activations"] += 1
                    if float(np.hypot(x_pred - fft_x, y_pred - fft_y)) > 5.0:
                        override_audit["fft_overrides"] += 1
                        if car_err <= 5.0 and fft_err > 5.0:
                            override_audit["correct_overrides"] += 1
                        elif car_err > 5.0 and fft_err <= 5.0:
                            override_audit["harmful_overrides"] += 1

            dt = (time.perf_counter() - t0) * 1000.0
            err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
            errors.append(err)
            latencies.append(dt)

            if diff not in per_diff:
                per_diff[diff] = []
            per_diff[diff].append(err)

        err_arr = np.array(errors)
        lat_arr = np.array(latencies)

        acc_1 = float(np.mean(err_arr <= 1.0)) * 100
        acc_3 = float(np.mean(err_arr <= 3.0)) * 100
        acc_5 = float(np.mean(err_arr <= 5.0)) * 100
        acc_10 = float(np.mean(err_arr <= 10.0)) * 100

        methods[method_name] = {
            "Acc_le1px": round(acc_1, 2),
            "Acc_le3px": round(acc_3, 2),
            "Acc_le5px": round(acc_5, 2),
            "Acc_le10px": round(acc_10, 2),
            "Mean_Err": round(float(np.mean(err_arr)), 2),
            "Median_Err": round(float(np.median(err_arr)), 2),
            "P95_Err": round(float(np.percentile(err_arr, 95)), 2),
            "Mean_Lat_ms": round(float(np.mean(lat_arr)), 2),
            "P95_Lat_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "per_diff": {d: round(float(np.mean(np.array(errs) <= 5.0)) * 100, 1)
                         for d, errs in per_diff.items()},
            "errors": err_arr,
        }

    print(f"\n{'='*95}")
    print(f"  BENCHMARK RESULTS: {label}")
    print(f"{'='*95}")
    header = f"{'Method':<20s} {'<=1px %':>8s} {'<=3px %':>8s} {'<=5px %':>8s} {'<=10px %':>8s} {'MeanErr':>8s} {'MedErr':>8s} {'P95Err':>8s} {'MeanLat':>9s} {'P95Lat':>9s}"
    print(header)
    print("-" * len(header))

    for method_name, stats in methods.items():
        if isinstance(stats, list):
            continue
        print(f"{method_name:<20s} {stats['Acc_le1px']:>7.2f}% {stats['Acc_le3px']:>7.2f}% "
              f"{stats['Acc_le5px']:>7.2f}% {stats['Acc_le10px']:>7.2f}% "
              f"{stats['Mean_Err']:>7.2f} {stats['Median_Err']:>7.2f} {stats['P95_Err']:>7.2f} "
              f"{stats['Mean_Lat_ms']:>8.2f} {stats['P95_Lat_ms']:>8.2f}")

    print(f"\n  Difficulty Breakdown (<=5px accuracy %):")
    for method_name, stats in methods.items():
        if isinstance(stats, list):
            continue
        diff_str = " | ".join(f"{d}: {v:.1f}%" for d, v in stats["per_diff"].items())
        print(f"    {method_name}: {diff_str}")

    print(f"\n  AI OVERRIDE AUDIT METRICS:")
    act = override_audit["pace_activations"]
    ovr = override_audit["fft_overrides"]
    cor = override_audit["correct_overrides"]
    har = override_audit["harmful_overrides"]
    override_rate = (cor / max(1, ovr)) * 100 if ovr > 0 else 0.0

    print(f"    - PACE Activations: {act}/{n} ({act/n*100:.1f}%)")
    print(f"    - Total FFT Overrides: {ovr}")
    print(f"    - Correct Overrides (FFT Wrong -> PACE Correct): {cor}")
    print(f"    - Harmful Overrides (FFT Correct -> PACE Wrong): {har}")
    print(f"    - Correct Override Rate: {override_rate:.1f}%")

    print(f"{'='*95}")
    return methods, override_audit


def main():
    os.makedirs("results", exist_ok=True)
    res_120, _ = evaluate_car_benchmark(ORIGINAL_MANIFEST, "Original 120-Case Benchmark")
    res_holdout, _ = evaluate_car_benchmark(HOLDOUT_MANIFEST, "Held-Out Test Set (Unseen Seeds)")

    if res_120 and res_holdout:
        rows = []
        for dataset_name, res_dict in [("Original_120", res_120), ("Holdout_Test", res_holdout)]:
            for method, stats in res_dict.items():
                if isinstance(stats, dict):
                    rows.append({
                        "Dataset": dataset_name,
                        "Method": method,
                        "Acc_le1px": stats["Acc_le1px"],
                        "Acc_le3px": stats["Acc_le3px"],
                        "Acc_le5px": stats["Acc_le5px"],
                        "Acc_le10px": stats["Acc_le10px"],
                        "Mean_Err": stats["Mean_Err"],
                        "Median_Err": stats["Median_Err"],
                        "P95_Err": stats["P95_Err"],
                        "Mean_Lat_ms": stats["Mean_Lat_ms"],
                        "P95_Lat_ms": stats["P95_Lat_ms"],
                    })
        df_out = pd.DataFrame(rows)
        csv_path = "results/car_benchmark_results.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\nSaved CAR benchmark results to '{csv_path}'")


if __name__ == "__main__":
    main()
