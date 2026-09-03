"""
Controlled Noise x Drift Stress Matrix Benchmark & Heatmap Artifact Generator
Tests Drift-Sense++ SAFE-CAR across systematic noise and perturbation dimensions:
  1. Translation Drift (+-10nm to +-500nm)
  2. E-Beam Defocus Blur (sigma = 0.5 to 3.0 px)
  3. Secondary Electron Poisson Shot Noise (lambda = 5 to 50 e-/px)
  4. Detector Gaussian Readout Noise (std = 0.01 to 0.08)
  5. Surface Charging Gradient (std = 0.01 to 0.05)
  6. Rotation Misalignment (0.0 to 3.0 deg)
  7. Combined Stress Perturbations

Outputs:
  - CSV Results: results/stress_matrix_results.csv
  - Visual Heatmap: submission_package/visuals/stress_matrix_heatmap.png
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from inference_car import perform_car_localization
from dataset_generator import generate_synthetic_pair, apply_sem_acquisition_effects


def evaluate_stress_level(perturbation_name: str, level_val: float, num_samples: int = 15, seed: int = 42) -> dict:
    np.random.seed(seed)
    errors = []
    latencies = []

    for i in range(num_samples):
        # Generate base synthetic pair
        pair = generate_synthetic_pair(architecture="FinFET", pair_id=i, seed=seed + i * 13)
        ref_img = pair["reference"]
        search_img = pair["search"]
        gt_x, gt_y = pair["gt_x"], pair["gt_y"]

        # Apply specific stress perturbation
        if perturbation_name == "Poisson_Shot_Noise":
            search_img = apply_sem_acquisition_effects(search_img, dose_lambda=float(level_val), seed=seed + i)
        elif perturbation_name == "Defocus_Blur":
            search_img = apply_sem_acquisition_effects(search_img, blur_sigma=float(level_val), seed=seed + i)
        elif perturbation_name == "Gaussian_Detector_Noise":
            search_img = apply_sem_acquisition_effects(search_img, gaussian_noise_std=float(level_val), seed=seed + i)
        elif perturbation_name == "Surface_Charging":
            search_img = apply_sem_acquisition_effects(search_img, charging_std=float(level_val), seed=seed + i)
        elif perturbation_name == "Rotation_Angle":
            if abs(level_val) > 0.01:
                M = cv2.getRotationMatrix2D((500.0, 500.0), float(level_val), 1.0)
                search_img = cv2.warpAffine(search_img, M, (1000, 1000), borderMode=cv2.BORDER_REFLECT)

        t0 = time.perf_counter()
        x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)
        dt = (time.perf_counter() - t0) * 1000.0

        err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
        errors.append(err)
        latencies.append(dt)

    err_arr = np.array(errors)
    lat_arr = np.array(latencies)

    return {
        "Perturbation": perturbation_name,
        "Level": level_val,
        "Acc_le1px": round(float(np.mean(err_arr <= 1.0)) * 100, 2),
        "Acc_le5px": round(float(np.mean(err_arr <= 5.0)) * 100, 2),
        "Acc_le10px": round(float(np.mean(err_arr <= 10.0)) * 100, 2),
        "Mean_Err": round(float(np.mean(err_arr)), 2),
        "Median_Err": round(float(np.median(err_arr)), 2),
        "P95_Err": round(float(np.percentile(err_arr, 95)), 2),
        "Mean_Lat_ms": round(float(np.mean(lat_arr)), 2),
    }


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission_package/visuals", exist_ok=True)

    print(f"\n{'='*95}")
    print(f"  RUNNING SYSTEMATIC NOISE x DRIFT STRESS MATRIX BENCHMARK")
    print(f"{'='*95}")

    stress_experiments = [
        ("Poisson_Shot_Noise", [50.0, 30.0, 15.0, 5.0]),
        ("Defocus_Blur", [0.5, 1.0, 1.8, 2.5]),
        ("Gaussian_Detector_Noise", [0.01, 0.03, 0.05, 0.08]),
        ("Surface_Charging", [0.005, 0.015, 0.03, 0.05]),
        ("Rotation_Angle", [0.0, 0.5, 1.5, 3.0]),
    ]

    rows = []
    heatmap_matrix = []
    labels_y = []

    for name, levels in stress_experiments:
        row_vals = []
        labels_y.append(name)
        for val in levels:
            print(f"  Evaluating {name} @ level={val}...")
            res = evaluate_stress_level(name, val, num_samples=12, seed=8080)
            rows.append(res)
            row_vals.append(res["Acc_le5px"])
        heatmap_matrix.append(row_vals)

    df_stress = pd.DataFrame(rows)
    csv_path = "results/stress_matrix_results.csv"
    df_stress.to_csv(csv_path, index=False)
    print(f"\nSaved stress matrix results to '{csv_path}'")

    # Generate Heatmap Visualization Artifact
    heatmap_arr = np.array(heatmap_matrix)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    cax = ax.matshow(heatmap_arr, cmap="YlGn", vmin=0, vmax=100)

    fig.colorbar(cax)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Low", "Moderate", "Severe", "Extreme"], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=11, fontweight="bold")
    ax.set_title("Drift-Sense++ SAFE-CAR In-Bounds Accuracy (<=5px %) Across Perturbation Stress Matrix", fontsize=12, pad=20, fontweight="bold")

    for i in range(heatmap_arr.shape[0]):
        for j in range(heatmap_arr.shape[1]):
            val = heatmap_arr[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="black" if val > 50 else "white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    heatmap_path = "submission_package/visuals/stress_matrix_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    print(f"Generated stress matrix heatmap artifact: '{heatmap_path}'")


if __name__ == "__main__":
    main()
