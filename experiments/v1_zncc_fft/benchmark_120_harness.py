"""
Comprehensive 120-Case Benchmark Harness & 7-Variant Ablation Study Engine
Evaluates:
- Baseline ZNCC & FFT-NCC
- Multi-Scale Dual V4
- Multi-Anchor Consensus Engine
- V5 Multi-Feature Structural Verification (Intensity + Gradient + Orientation + Frequency Pitch + Phase)
- Full Drift-Sense V5 Architecture
Logs accuracy (<=1px, <=3px, <=5px, <=10px), mean/median/P95 error, runtime P95, and difficulty breakdowns.
"""

import os
import sys
import time
import json
import csv
import numpy as np
import cv2
import pandas as pd

from inference_v5 import (
    perform_drift_sense_v5,
    normalize_intensity,
    extract_gradient_map,
    extract_distinct_spatial_peaks,
    local_phase_correlation,
    compute_frequency_pitch_spectrum,
    subpixel_refine_2d
)
from anchor_consensus import retrieve_multi_anchor_consensus
from inference import perform_proposed_localization as perform_v4_localization
from hf_space.baseline_solution.zncc import zncc_match

DATASET_DIR = "data/benchmark_120"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")


def run_fft_ncc(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, float]:
    ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search_img.astype(np.float32), ref_down.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr)
    return float(max_loc[0] + 50), float(max_loc[1] + 50), float(max_val)


def run_variant(ref_img: np.ndarray, search_img: np.ndarray, variant_id: str) -> tuple[float, float, float, dict]:
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0

    # Variant 0: Baseline ZNCC
    if variant_id == "V0_ZNCC":
        res = zncc_match(ref_img, search_img)
        return float(res["x"]), float(res["y"]), float(res["score"]), res

    # Variant A: Plain FFT-NCC
    if variant_id == "Var_A_FFT_NCC":
        x, y, score = run_fft_ncc(ref_img, search_img)
        return x, y, score, {"score": score}

    # Variant B: FFT + Local Verification
    if variant_id == "Var_B_FFT_LocalVerif":
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        ref_n = normalize_intensity(ref_down)
        ref_g, _ = extract_gradient_map(ref_down)
        search_n = normalize_intensity(search_img)
        search_g, _ = extract_gradient_map(search_img)

        c_i = cv2.matchTemplate(search_n, ref_n, cv2.TM_CCOEFF_NORMED)
        peaks = extract_distinct_spatial_peaks(c_i, top_k=6, min_dist=15)
        cands = []
        for p in peaks:
            cx, cy = p["x"] + 50, p["y"] + 50
            y1, y2 = max(0, cy - 50), min(sh, cy + 50)
            x1, x2 = max(0, cx - 50), min(sw, cx + 50)
            sp_n = search_n[y1:y2, x1:x2]
            sp_g = search_g[y1:y2, x1:x2]
            if sp_n.shape != (100, 100):
                sp_n = cv2.resize(sp_n, (100, 100))
                sp_g = cv2.resize(sp_g, (100, 100))
            _, _, ps = local_phase_correlation(ref_n, sp_n)
            g_corr = float(np.corrcoef(ref_g.ravel(), sp_g.ravel())[0, 1])
            score = 0.55 * p["score"] + 0.30 * g_corr + 0.15 * max(0.0, ps)
            cands.append({"x": cx, "y": cy, "score": score, "peak_x": p["x"], "peak_y": p["y"], "corr_plane": c_i})

        cands.sort(key=lambda c: c["score"], reverse=True)
        best = cands[0]
        sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
        return float(sub_x + 50), float(sub_y + 50), best["score"], best

    # Variant C: FFT + Periodicity Rule
    if variant_id == "Var_C_FFT_Periodicity":
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        corr = cv2.matchTemplate(search_img.astype(np.float32), ref_down.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        peaks = extract_distinct_spatial_peaks(corr, top_k=6, min_dist=15)
        cands = [{"x": p["x"] + 50, "y": p["y"] + 50, "score": p["score"], "peak_x": p["x"], "peak_y": p["y"], "corr_plane": corr,
                  "dist": np.hypot(p["x"] + 50 - search_cx, p["y"] + 50 - search_cy)} for p in peaks]
        cands.sort(key=lambda c: c["score"], reverse=True)
        best = cands[0]
        if len(cands) > 1 and (cands[0]["score"] - cands[1]["score"]) < 0.035:
            pool = [c for c in cands if (cands[0]["score"] - c["score"]) < 0.035]
            best = min(pool, key=lambda c: c["dist"])
        sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
        return float(sub_x + 50), float(sub_y + 50), best["score"], best

    # Variant D: FFT + Local Verification + Periodicity
    if variant_id == "Var_D_FFT_Verif_Periodicity":
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        ref_n = normalize_intensity(ref_down)
        ref_g, _ = extract_gradient_map(ref_down)
        search_n = normalize_intensity(search_img)
        search_g, _ = extract_gradient_map(search_img)

        c_i = cv2.matchTemplate(search_n, ref_n, cv2.TM_CCOEFF_NORMED)
        peaks = extract_distinct_spatial_peaks(c_i, top_k=6, min_dist=15)
        cands = []
        for p in peaks:
            cx, cy = p["x"] + 50, p["y"] + 50
            y1, y2 = max(0, cy - 50), min(sh, cy + 50)
            x1, x2 = max(0, cx - 50), min(sw, cx + 50)
            sp_n = search_n[y1:y2, x1:x2]
            sp_g = search_g[y1:y2, x1:x2]
            if sp_n.shape != (100, 100):
                sp_n = cv2.resize(sp_n, (100, 100))
                sp_g = cv2.resize(sp_g, (100, 100))
            _, _, ps = local_phase_correlation(ref_n, sp_n)
            g_corr = float(np.corrcoef(ref_g.ravel(), sp_g.ravel())[0, 1])
            score = 0.55 * p["score"] + 0.30 * g_corr + 0.15 * max(0.0, ps)
            dist = np.hypot(cx - search_cx, cy - search_cy)
            cands.append({"x": cx, "y": cy, "score": score, "dist": dist, "peak_x": p["x"], "peak_y": p["y"], "corr_plane": c_i})

        cands.sort(key=lambda c: c["score"], reverse=True)
        best = cands[0]
        if len(cands) > 1 and (cands[0]["score"] - cands[1]["score"]) < 0.035:
            pool = [c for c in cands if (cands[0]["score"] - c["score"]) < 0.035]
            best = min(pool, key=lambda c: c["dist"])
        sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
        return float(sub_x + 50), float(sub_y + 50), best["score"], best

    # Variant E: Multi-Anchor Consensus Retrieval Only
    if variant_id == "Var_E_Anchor_Consensus":
        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        consensus_cands, _ = retrieve_multi_anchor_consensus(ref_100, search_img, scales=(1.0,), top_k_candidates=5)
        best = consensus_cands[0]
        return float(best["center_x"]), float(best["center_y"]), float(best["consensus_score"]), best

    # Variant F: Multi-Anchor + Verification + Periodicity
    if variant_id == "Var_F_Anchor_Verif_Periodicity":
        x, y, meta = perform_drift_sense_v5(ref_img, search_img, scales=(1.00,), rotations=(0.0,), use_multi_anchor=True)
        return x, y, meta["score"], meta

    # Variant G: Full Drift-Sense V5 (Multi-Anchor + Scale/Rotation Search + Orientation/Pitch Spectrum)
    if variant_id == "Var_G_V5_Full":
        x, y, meta = perform_drift_sense_v5(ref_img, search_img, scales=(0.97, 1.00, 1.03), rotations=(-2.0, 0.0, 2.0), use_multi_anchor=True)
        return x, y, meta["score"], meta

    # V4 Proposed Baseline
    if variant_id == "V4_Proposed_Baseline":
        x, y, meta = perform_v4_localization(ref_img, search_img, verbose=False)
        return x, y, meta.get("score", 0.0), meta

    return 500.0, 500.0, 0.0, {}


def main():
    print("=" * 95)
    print("       DRIFT-SENSE 120-CASE BENCHMARK & 7-VARIANT ABLATION MATRIX (V5)       ")
    print("=" * 95)

    df_manifest = pd.read_csv(MANIFEST_PATH)

    variants = [
        "V0_ZNCC",
        "Var_A_FFT_NCC",
        "Var_B_FFT_LocalVerif",
        "Var_C_FFT_Periodicity",
        "Var_D_FFT_Verif_Periodicity",
        "Var_E_Anchor_Consensus",
        "Var_F_Anchor_Verif_Periodicity",
        "Var_G_V5_Full",
        "V4_Proposed_Baseline"
    ]

    results = []
    diff_breakdown = {}

    for v_id in variants:
        print(f"\n--> Running Variant: [{v_id}]...")
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
            x_pred, y_pred, score, meta = run_variant(ref_img, search_img, v_id)
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
    print("                     FULL 120-CASE ABLATION STUDY RESULTS                     ")
    print("=" * 95)
    print(df_res.to_string(index=False))
    print("=" * 95)

    print("\nAccuracy (<=5px) Breakdown by Difficulty Level (%):")
    df_diff = pd.DataFrame(diff_breakdown).T
    print(df_diff.to_string())
    print("=" * 95)

    os.makedirs("results", exist_ok=True)
    out_csv = "results/benchmark_120_v5_ablation_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nWrote full V5 benchmark ablation summary to '{out_csv}'")


if __name__ == "__main__":
    main()
