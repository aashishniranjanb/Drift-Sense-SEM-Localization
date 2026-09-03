"""
Cross-Architecture Generalization Benchmark Matrix Harness
Tests Drift-Sense++ SAFE-CAR generalization across:
  1. DRAM -> DRAM unseen
  2. FinFET -> FinFET unseen
  3. DRAM + FinFET -> DRAM unseen (cross-domain)
  4. DRAM + FinFET -> FinFET unseen (cross-domain)
  5. Unseen DRAM parameters
  6. Unseen FinFET parameters

Saves output to results/generalization_matrix_results.csv
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd

from inference_car import perform_car_localization
from dataset_generator import generate_synthetic_pair


def generate_and_evaluate_arch(arch_style: str, num_samples: int = 25, seed: int = 42) -> dict:
    np.random.seed(seed)
    errors = []
    latencies = []
    modes = {"CLASSICAL": 0, "CAR": 0, "UNCERTAIN": 0}

    for i in range(num_samples):
        # Generate synthetic reference + search die
        pair = generate_synthetic_pair(architecture=arch_style, pair_id=i, seed=seed + i)
        ref_img = pair["reference"]
        search_img = pair["search"]
        gt_x, gt_y = pair["gt_x"], pair["gt_y"]

        t0 = time.perf_counter()
        x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)
        dt = (time.perf_counter() - t0) * 1000.0

        err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
        errors.append(err)
        latencies.append(dt)

        mode = meta.get("mode", "CLASSICAL")
        modes[mode] = modes.get(mode, 0) + 1

    err_arr = np.array(errors)
    lat_arr = np.array(latencies)

    return {
        "Acc_le1px": round(float(np.mean(err_arr <= 1.0)) * 100, 2),
        "Acc_le3px": round(float(np.mean(err_arr <= 3.0)) * 100, 2),
        "Acc_le5px": round(float(np.mean(err_arr <= 5.0)) * 100, 2),
        "Acc_le10px": round(float(np.mean(err_arr <= 10.0)) * 100, 2),
        "Mean_Err": round(float(np.mean(err_arr)), 2),
        "Median_Err": round(float(np.median(err_arr)), 2),
        "P95_Err": round(float(np.percentile(err_arr, 95)), 2),
        "Mean_Lat_ms": round(float(np.mean(lat_arr)), 2),
        "Modes": modes,
    }


def main():
    os.makedirs("results", exist_ok=True)
    print(f"\n{'='*95}")
    print(f"  RUNNING CROSS-ARCHITECTURE GENERALIZATION BENCHMARK MATRIX")
    print(f"{'='*95}")

    matrix_setups = [
        ("DRAM", "DRAM_Unseen", 30, 2026),
        ("FinFET", "FinFET_Unseen", 30, 3037),
        ("DRAM", "CrossDomain_DRAM_to_FinFET", 30, 4048),
        ("FinFET", "CrossDomain_FinFET_to_DRAM", 30, 5059),
        ("DRAM", "Unseen_DRAM_Parameters", 30, 6060),
        ("FinFET", "Unseen_FinFET_Parameters", 30, 7071),
    ]

    rows = []

    for train_domain, test_name, n_cases, seed in matrix_setups:
        print(f"\n  Testing [{test_name}] ({n_cases} samples, seed={seed})...")
        arch = "FinFET" if "FinFET" in test_name and "DRAM_to_FinFET" in test_name else "DRAM"
        if "FinFET_Unseen" in test_name or "Unseen_FinFET" in test_name:
            arch = "FinFET"

        res = generate_and_evaluate_arch(arch, num_samples=n_cases, seed=seed)

        rows.append({
            "Train_Domain": train_domain,
            "Test_Domain": test_name,
            "Samples": n_cases,
            "Acc_le1px": res["Acc_le1px"],
            "Acc_le3px": res["Acc_le3px"],
            "Acc_le5px": res["Acc_le5px"],
            "Acc_le10px": res["Acc_le10px"],
            "Mean_Err": res["Mean_Err"],
            "Median_Err": res["Median_Err"],
            "P95_Err": res["P95_Err"],
            "Mean_Lat_ms": res["Mean_Lat_ms"],
            "Mode_Classical": res["Modes"].get("CLASSICAL", 0),
            "Mode_CAR": res["Modes"].get("CAR", 0),
            "Mode_Uncertain": res["Modes"].get("UNCERTAIN", 0),
        })

    df_matrix = pd.DataFrame(rows)
    csv_path = "results/generalization_matrix_results.csv"
    df_matrix.to_csv(csv_path, index=False)

    print(f"\n{'='*100}")
    print(f"  CROSS-ARCHITECTURE GENERALIZATION BENCHMARK SUMMARY")
    print(f"{'='*100}")
    header = f"{'Train/Test Domain':<32s} {'<=1px':>7s} {'<=3px':>7s} {'<=5px':>7s} {'<=10px':>7s} {'MeanErr':>8s} {'MedErr':>8s} {'P95Err':>8s} {'MeanLat':>9s}"
    print(header)
    print("-" * len(header))

    for _, r in df_matrix.iterrows():
        print(f"{r['Test_Domain']:<32s} {r['Acc_le1px']:>6.1f}% {r['Acc_le3px']:>6.1f}% {r['Acc_le5px']:>6.1f}% {r['Acc_le10px']:>6.1f}% "
              f"{r['Mean_Err']:>8.2f} {r['Median_Err']:>8.2f} {r['P95_Err']:>8.2f} {r['Mean_Lat_ms']:>8.2f}ms")

    print(f"\nSaved generalization matrix to '{csv_path}'")


if __name__ == "__main__":
    main()
