"""
Comprehensive 120-Case Benchmark Harness & Ablation Study Engine
Evaluates Baselines (ZNCC, FFT-NCC), Structural Representations (Intensity, Gradient, PC, Hybrid),
and the Incremental Drift-Sense++ Pipeline with Failure Taxonomy & Profiling.
"""

import os
import sys
import time
import json
import csv
import numpy as np
import cv2
import pandas as pd

from inference import (
    compute_structural_map,
    compute_fast_phase_congruency,
    compute_psr,
    perform_drift_sense_localization
)
from hf_space.baseline_solution.zncc import zncc_match

DATASET_DIR = "data/benchmark_120"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")


def fft_cross_correlation(search_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    """Computes 2D normalized cross-correlation map via OpenCV matchTemplate."""
    return cv2.matchTemplate(search_img.astype(np.float32), ref_img.astype(np.float32), cv2.TM_CCOEFF_NORMED)


def run_fft_ncc(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, float]:
    """Plain FFT correlation baseline without structural mapping or phase congruency."""
    ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    corr = fft_cross_correlation(search_img, ref_down)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    x_pred = float(max_loc[0] + 50)
    y_pred = float(max_loc[1] + 50)
    psr = compute_psr(corr, max_loc[0], max_loc[1])
    return x_pred, y_pred, psr


def extract_gradient_map(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    if g_mag.max() > 0:
        g_mag /= g_mag.max()
    return g_mag.astype(np.float32)


def run_pipeline_variant(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    variant_id: str
) -> tuple[float, float, float, str]:
    """Executes specific ablation stage variant."""
    # V0: Baseline ZNCC
    if variant_id == "V0_ZNCC":
        res = zncc_match(ref_img, search_img)
        return float(res["x"]), float(res["y"]), float(res["score"]), "OK"

    # V1: Baseline FFT-NCC
    if variant_id == "V1_FFT_NCC":
        x, y, psr = run_fft_ncc(ref_img, search_img)
        return x, y, psr, "OK"

    # V2: FFT + Gradient
    if variant_id == "V2_FFT_Grad":
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        g_ref = extract_gradient_map(ref_down)
        g_search = extract_gradient_map(search_img)
        corr = fft_cross_correlation(g_search, g_ref)
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)
        return float(max_loc[0] + 50), float(max_loc[1] + 50), compute_psr(corr, max_loc[0], max_loc[1]), "OK"

    # V3: FFT + Hybrid (PC + Grad)
    if variant_id == "V3_FFT_Hybrid":
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        h_ref, _ = compute_structural_map(ref_down, 10.0)
        h_search, _ = compute_structural_map(search_img, 20.0)
        corr = fft_cross_correlation(h_search, h_ref)
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)
        return float(max_loc[0] + 50), float(max_loc[1] + 50), compute_psr(corr, max_loc[0], max_loc[1]), "OK"

    # V4: + Scale/Rotation Search
    if variant_id == "V4_Scale_Rot":
        best_val = -1.0
        best_coord = (500.0, 500.0)
        h_search, _ = compute_structural_map(search_img, 20.0)

        scales = [0.98, 1.0, 1.02]
        rotations = [-2.0, 0.0, 2.0]

        for s in scales:
            w_scaled = max(10, int(round(100 * s)))
            h_scaled = max(10, int(round(100 * s)))
            ref_scaled = cv2.resize(ref_img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)

            for r in rotations:
                if r != 0.0:
                    M = cv2.getRotationMatrix2D((w_scaled / 2.0, h_scaled / 2.0), r, 1.0)
                    ref_rot = cv2.warpAffine(ref_scaled, M, (w_scaled, h_scaled))
                else:
                    ref_rot = ref_scaled

                h_ref, _ = compute_structural_map(ref_rot, 10.0)
                corr = fft_cross_correlation(h_search, h_ref)
                _, max_val, _, max_loc = cv2.minMaxLoc(corr)
                if max_val > best_val:
                    best_val = max_val
                    best_coord = (float(max_loc[0] + w_scaled / 2.0), float(max_loc[1] + h_scaled / 2.0))

        return best_coord[0], best_coord[1], best_val, "OK"

    # V10: Full Drift-Sense++ with Subpixel Refinement
    x, y, meta = perform_drift_sense_localization(ref_img, search_img, verbose=False)
    return x, y, meta["psr"], meta["status"]


def main():
    print("Loading 120-case manifest...")
    df_manifest = pd.read_csv(MANIFEST_PATH)

    variants = [
        "V0_ZNCC",
        "V1_FFT_NCC",
        "V2_FFT_Grad",
        "V3_FFT_Hybrid",
        "V4_Scale_Rot",
        "V10_Drift_Sense_Plus_Plus"
    ]

    results = []

    print("\nRunning Benchmark Evaluation across 120 samples...")

    for v_id in variants:
        print(f"\nEvaluating Variant: [{v_id}]...")
        t_start = time.perf_counter()

        errors = []
        accuracies_1px = []
        accuracies_3px = []
        accuracies_5px = []

        for idx, row in df_manifest.iterrows():
            ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)

            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

            x_pred, y_pred, score, status = run_pipeline_variant(ref_img, search_img, v_id)

            err = float(np.sqrt((x_pred - gt_x)**2 + (y_pred - gt_y)**2))
            errors.append(err)

            accuracies_1px.append(err <= 1.0)
            accuracies_3px.append(err <= 3.0)
            accuracies_5px.append(err <= 5.0)

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        avg_latency = t_elapsed / len(df_manifest)

        errors_arr = np.array(errors)
        results.append({
            "Variant": v_id,
            "Acc (<=1px) %": round(float(np.mean(accuracies_1px)) * 100, 2),
            "Acc (<=3px) %": round(float(np.mean(accuracies_3px)) * 100, 2),
            "Acc (<=5px) %": round(float(np.mean(accuracies_5px)) * 100, 2),
            "Mean Err (px)": round(float(np.mean(errors_arr)), 2),
            "Median Err (px)": round(float(np.median(errors_arr)), 2),
            "P95 Err (px)": round(float(np.percentile(errors_arr, 95)), 2),
            "Mean Latency (ms)": round(avg_latency, 2)
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                     DRIFT-SENSE 120-CASE BENCHMARK RESULTS                     ")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    res_csv = "results/benchmark_120_ablation_results.csv"
    os.makedirs("results", exist_ok=True)
    df_res.to_csv(res_csv, index=False)
    print(f"\nWrote full benchmark ablation summary to '{res_csv}'")


if __name__ == "__main__":
    main()
