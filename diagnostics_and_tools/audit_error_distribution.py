"""
Audit Error Distribution & Localization CDF Generator
Checks exact error histograms across bins [0-1], (1-2], (2-3], (3-4], (4-5], (5-10], (10-25], (25-50], (50-100], >100 px
to verify accuracy metric behavior and plot cumulative distribution function (CDF).
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd

from inference import (
    extract_gradient_map,
    normalize_intensity,
    perform_proposed_localization
)
from hf_space.baseline_solution.zncc import zncc_match

MANIFEST_PATH = "data/benchmark_120/manifest.csv"

def run_fft_ncc(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float]:
    ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search_img.astype(np.float32), ref_down.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr)
    return float(max_loc[0] + 50), float(max_loc[1] + 50)


def audit_errors():
    print("=" * 80)
    print("          AUDITING EXACT ERROR HISTOGRAMS & CDF METRICS (120 SAMPLES)          ")
    print("=" * 80)

    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: manifest '{MANIFEST_PATH}' not found!")
        return

    df_manifest = pd.read_csv(MANIFEST_PATH)
    variants = ["V0_ZNCC", "V1_FFT_NCC", "V3_MultiScale_Dual", "V4_Proposed_Architecture"]

    bins = [0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 10000]
    bin_labels = ["0-1 px", "1-2 px", "2-3 px", "3-4 px", "4-5 px", "5-10 px", "10-25 px", "25-50 px", "50-100 px", ">100 px"]

    results_by_variant = {}

    for v_id in variants:
        errors = []
        for idx, row in df_manifest.iterrows():
            ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

            if v_id == "V0_ZNCC":
                res = zncc_match(ref_img, search_img)
                x_pred, y_pred = float(res["x"]), float(res["y"])
            elif v_id == "V1_FFT_NCC":
                x_pred, y_pred = run_fft_ncc(ref_img, search_img)
            elif v_id == "V3_MultiScale_Dual":
                search_norm = normalize_intensity(search_img)
                search_grad = extract_gradient_map(search_img)
                best_score = -1.0
                best_x, best_y = 500.0, 500.0
                for scale in [0.95, 0.98, 1.00, 1.02, 1.05]:
                    tw = max(10, int(round(100 * scale)))
                    th = max(10, int(round(100 * scale)))
                    ref_s = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
                    r_norm = normalize_intensity(ref_s)
                    r_grad = extract_gradient_map(ref_s)
                    c_i = cv2.matchTemplate(search_norm, r_norm, cv2.TM_CCOEFF_NORMED)
                    c_g = cv2.matchTemplate(search_grad, r_grad, cv2.TM_CCOEFF_NORMED)
                    c_combo = 0.55 * c_i + 0.45 * c_g
                    _, max_val, _, max_loc = cv2.minMaxLoc(c_combo)
                    if max_val > best_score:
                        best_score = max_val
                        best_x = max_loc[0] + tw / 2.0
                        best_y = max_loc[1] + th / 2.0
                x_pred, y_pred = best_x, best_y
            elif v_id == "V4_Proposed_Architecture":
                x_pred, y_pred, _ = perform_proposed_localization(ref_img, search_img, verbose=False)

            err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
            errors.append(err)

        errors_arr = np.array(errors)
        counts, _ = np.histogram(errors_arr, bins=bins)
        cdf = np.cumsum(counts) / len(errors) * 100

        results_by_variant[v_id] = {
            "errors": errors_arr,
            "counts": counts,
            "cdf": cdf
        }

    # Print Histogram Table
    df_hist = pd.DataFrame(index=bin_labels)
    for v_id in variants:
        df_hist[v_id + " Count"] = results_by_variant[v_id]["counts"]

    print("\nExact Error Count Histogram across 10 Bins:")
    print(df_hist.to_string())

    print("\n" + "-" * 80)
    print("Cumulative Distribution Function (CDF) P(error <= e) %:")
    df_cdf = pd.DataFrame(index=["<=1px", "<=2px", "<=3px", "<=4px", "<=5px", "<=10px", "<=25px", "<=50px", "<=100px", "Total"])
    for v_id in variants:
        cdf_vals = list(np.round(results_by_variant[v_id]["cdf"], 2))
        df_cdf[v_id] = cdf_vals

    print(df_cdf.to_string())
    print("=" * 80)

    # Detailed inspection of V4 errors between 0 and 10 px
    v4_errs = results_by_variant["V4_Proposed_Architecture"]["errors"]
    small_v4 = v4_errs[v4_errs <= 10.0]
    print(f"\nV4 Specific Breakdown of Small Errors (<=10 px):")
    print(f"Total samples <= 10 px: {len(small_v4)}")
    print(f"Sample values <= 10 px: {np.round(np.sort(small_v4), 4)}")

    os.makedirs("results", exist_ok=True)
    df_cdf.to_csv("results/cdf_error_distribution.csv")
    print("\nWrote CDF error distribution to 'results/cdf_error_distribution.csv'")

if __name__ == "__main__":
    audit_errors()
