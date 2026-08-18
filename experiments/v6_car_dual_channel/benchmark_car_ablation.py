"""
Comprehensive 5-Variant Ablation Study on Frozen 200-Case Held-Out Test Set
Evaluates:
  Variant A: Intensity FFT alone (Baseline)
  Variant B: Gradient FFT alone (Structural retrieval)
  Variant C: Intensity union Gradient (Dual-Channel Candidate Union without AI)
  Variant D: Intensity union Gradient + CAR Confidence Gate + Subpixel Consensus (Final)
  Variant E: Variant D + RGB Support (Bonus Branch)

Reports:
  - Retrieval Recall: Top-1, Top-5, Top-10, Top-20
  - Localization Accuracy: <=1px, <=3px, <=5px, <=10px, <=25px
  - Error Metrics: Mean Error, Median Error, P95 Error
  - Safety & Override Audit: Total Overrides, Beneficial Overrides, Harmful Overrides, % entering CAR
  - Runtime: Mean Latency, P95 Latency, Max Latency
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd
import torch

from pace_model import ProcessAwareContextEncoder
from generate_pace_dataset import extract_directional_overlaps, extract_patch_safe, normalize_intensity
from inference_car import (
    compute_psr,
    estimator_a_phase_correlation,
    estimator_b_paraboloid_fit,
    evaluate_estimator_consensus,
    load_pace_model,
    extract_gradient
)

HOLDOUT_MANIFEST = "data/hcr_test/manifest.csv"
BENCHMARK_120_MANIFEST = "data/benchmark_120/manifest.csv"


# ─── Retrieval Helpers ───────────────────────────────────────────────────

def extract_candidates_single_map(corr_plane: np.ndarray, k_max: int = 10, offset: float = 50.0) -> list[dict]:
    work = corr_plane.copy()
    ch, cw = work.shape
    candidates = []
    for _ in range(k_max):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + offset
        cy = py + offset
        candidates.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "score": float(max_val),
            "corr_plane": corr_plane,
        })
        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0
    return candidates


def extract_dual_channel_union(search_img: np.ndarray, ref_img: np.ndarray, k_per_channel: int = 10) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Computes Intensity FFT and Gradient FFT independently, extracts Top-10 spatial peaks from each,
    and returns their Spatial Union (deduplicated within 12 px radius).
    """
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)

    search_norm = normalize_intensity(search_proc)
    ref_100_norm = normalize_intensity(cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA))

    search_grad = extract_gradient(search_proc)
    ref_100_grad = extract_gradient(cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA))

    # Channel A: Intensity FFT-NCC
    c_i = cv2.matchTemplate(search_norm, ref_100_norm, cv2.TM_CCOEFF_NORMED)
    # Channel B: Gradient FFT-NCC
    c_g = cv2.matchTemplate(search_grad, ref_100_grad, cv2.TM_CCOEFF_NORMED)

    cands_i = extract_candidates_single_map(c_i, k_max=k_per_channel)
    for c in cands_i:
        c["source"] = "intensity"
        c["i_score"] = c["score"]
        c["g_score"] = float(c_g[c["peak_y"], c["peak_x"]]) if c["peak_y"] < c_g.shape[0] and c["peak_x"] < c_g.shape[1] else 0.0

    cands_g = extract_candidates_single_map(c_g, k_max=k_per_channel)
    for c in cands_g:
        c["source"] = "gradient"
        c["g_score"] = c["score"]
        c["i_score"] = float(c_i[c["peak_y"], c["peak_x"]]) if c["peak_y"] < c_i.shape[0] and c["peak_x"] < c_i.shape[1] else 0.0

    # Spatial Union (Union preservation)
    union_cands = list(cands_i)
    for cg in cands_g:
        if not any(np.hypot(cg["cx"] - u["cx"], cg["cy"] - u["cy"]) < 12.0 for u in union_cands):
            union_cands.append(cg)

    # Sort union by highest combined feature score
    for c in union_cands:
        c["union_score"] = max(c["i_score"], c["g_score"])

    union_cands.sort(key=lambda c: c["union_score"], reverse=True)
    return union_cands[:20], c_i, c_g


# ─── Variant Runners ─────────────────────────────────────────────────────

def run_variant_a(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, list[dict]]:
    """Variant A: Intensity FFT alone."""
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    c_i = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    cands = extract_candidates_single_map(c_i, k_max=20)
    top = cands[0]
    sub_x, sub_y = estimator_b_paraboloid_fit(c_i, top["peak_x"], top["peak_y"])
    return float(sub_x + 50.0), float(sub_y + 50.0), cands


def run_variant_b(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, list[dict]]:
    """Variant B: Gradient FFT alone."""
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    search_grad = extract_gradient(search_img)
    ref_grad = extract_gradient(ref_100)
    c_g = cv2.matchTemplate(search_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    cands = extract_candidates_single_map(c_g, k_max=20)
    top = cands[0]
    sub_x, sub_y = estimator_b_paraboloid_fit(c_g, top["peak_x"], top["peak_y"])
    return float(sub_x + 50.0), float(sub_y + 50.0), cands


def run_variant_c(ref_img: np.ndarray, search_img: np.ndarray) -> tuple[float, float, list[dict]]:
    """Variant C: Intensity union Gradient without AI."""
    cands, c_i, _ = extract_dual_channel_union(search_img, ref_img)
    top = cands[0]
    sub_x, sub_y = estimator_b_paraboloid_fit(c_i, top["peak_x"], top["peak_y"])
    return float(sub_x + 50.0), float(sub_y + 50.0), cands


def run_variant_d(ref_img: np.ndarray, search_img: np.ndarray, model, device) -> tuple[float, float, list[dict], dict]:
    """Variant D: Dual-Channel Union + CAR Confidence Gate + Subpixel Consensus."""
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    cands, c_i, _ = extract_dual_channel_union(search_img, ref_img)
    if len(cands) == 0:
        return search_cx, search_cy, [], {"pace_activated": False}

    top1 = cands[0]
    top2_score = cands[1]["union_score"] if len(cands) > 1 else 0.0
    delta_s = top1["union_score"] - top2_score
    psr = compute_psr(c_i, top1["peak_x"], top1["peak_y"])

    # Strict Confidence Gate: Lock FFT Candidate #1 when unambiguous
    is_high_confidence = (delta_s >= 0.010 and psr >= 5.5) or (top1["union_score"] >= 0.85)

    if is_high_confidence or model is None:
        final_x, final_y, D, is_conf = evaluate_estimator_consensus(ref_100, search_img, top1)
        return final_x, final_y, cands, {"pace_activated": False, "consensus_D": D}

    # Ambiguous: Run PACE Residual Ranking
    ref_64 = normalize_intensity(cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA))
    ref_128 = normalize_intensity(cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA))
    ref_ovl = extract_directional_overlaps(ref_img, 500, 500, 32, offset=200)

    ref_64_t = torch.from_numpy(ref_64).unsqueeze(0).unsqueeze(0).to(device)
    ref_128_t = torch.from_numpy(ref_128).unsqueeze(0).unsqueeze(0).to(device)
    ref_ovl_t = torch.from_numpy(ref_ovl).unsqueeze(0).to(device)

    cand_64_list, cand_128_list, cand_ovl_list, cand_ncc_list = [], [], [], []
    for c in cands:
        p64 = normalize_intensity(extract_patch_safe(search_img, c["cx"], c["cy"], 64))
        p128 = normalize_intensity(extract_patch_safe(search_img, c["cx"], c["cy"], 128))
        povl = extract_directional_overlaps(search_img, c["cx"], c["cy"], 32, offset=40)
        cand_64_list.append(torch.from_numpy(p64).unsqueeze(0))
        cand_128_list.append(torch.from_numpy(p128).unsqueeze(0))
        cand_ovl_list.append(torch.from_numpy(povl))
        cand_ncc_list.append(c["union_score"])

    cand_64_batch = torch.stack(cand_64_list).to(device)
    cand_128_batch = torch.stack(cand_128_list).to(device)
    cand_ovl_batch = torch.stack(cand_ovl_list).to(device)
    cand_ncc_batch = torch.tensor(cand_ncc_list, dtype=torch.float32).to(device)

    with torch.no_grad():
        z_ref = model.forward_encoder(ref_64_t, ref_128_t, ref_ovl_t)
        z_cands = model.forward_encoder(cand_64_batch, cand_128_batch, cand_ovl_batch)
        scores = model(z_ref, z_cands, cand_ncc_batch).cpu().numpy()[0]

    for i, c in enumerate(cands):
        c["pace_score"] = float(scores[i])
        c["final_score"] = c["union_score"] + 0.08 * float(scores[i])

    cands.sort(key=lambda c: c["final_score"], reverse=True)
    top3 = cands[:3]
    for c in top3:
        c["dist_to_center"] = float(np.hypot(c["cx"] - search_cx, c["cy"] - search_cy))

    best = top3[0]
    if len(top3) >= 2:
        final_delta = top3[0]["final_score"] - top3[1]["final_score"]
        cand_dist = np.hypot(top3[0]["cx"] - top3[1]["cx"], top3[0]["cy"] - top3[1]["cy"])
        if final_delta < 0.005 and 15.0 <= cand_dist <= 120.0:
            ambiguity_pool = [c for c in top3 if (top3[0]["final_score"] - c["final_score"]) < 0.005]
            best = min(ambiguity_pool, key=lambda c: c["dist_to_center"])

    final_x, final_y, D, is_conf = evaluate_estimator_consensus(ref_100, search_img, best)
    return final_x, final_y, cands, {"pace_activated": True, "consensus_D": D}


def run_variant_e(ref_img: np.ndarray, search_img: np.ndarray, model, device) -> tuple[float, float, list[dict], dict]:
    """Variant E: Variant D with RGB / multi-channel luminance support."""
    if len(ref_img.shape) == 3 and ref_img.shape[2] == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img

    if len(search_img.shape) == 3 and search_img.shape[2] == 3:
        search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_img

    return run_variant_d(ref_gray, search_gray, model, device)


# ─── Main Ablation Harness ───────────────────────────────────────────────

def evaluate_ablation_on_manifest(manifest_path: str, label: str):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest '{manifest_path}' not found!")
        return None

    df = pd.read_csv(manifest_path)
    n = len(df)
    print(f"\n{'='*95}")
    print(f"  RUNNING DUAL-CHANNEL RETRIEVAL ABLATION STUDY: {label} ({n} samples)")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*95}")

    model, device = load_pace_model()

    variants = ["Variant_A_Intensity_FFT", "Variant_B_Gradient_FFT", "Variant_C_Dual_Channel_Union",
                "Variant_D_CAR_Final", "Variant_E_CAR_RGB_Bonus"]

    results = {}

    for var in variants:
        print(f"\n  Evaluating [{var}]...")
        errors = []
        latencies = []
        top1_recalls = []
        top5_recalls = []
        top10_recalls = []
        top20_recalls = []

        pace_activations = 0
        total_overrides = 0
        beneficial_overrides = 0
        harmful_overrides = 0

        for idx, row in df.iterrows():
            ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

            t0 = time.perf_counter()

            if var == "Variant_A_Intensity_FFT":
                px, py, cands = run_variant_a(ref_img, search_img)
            elif var == "Variant_B_Gradient_FFT":
                px, py, cands = run_variant_b(ref_img, search_img)
            elif var == "Variant_C_Dual_Channel_Union":
                px, py, cands = run_variant_c(ref_img, search_img)
            elif var == "Variant_D_CAR_Final":
                px, py, cands, meta = run_variant_d(ref_img, search_img, model, device)
                if meta.get("pace_activated", False):
                    pace_activations += 1
                    # Check override relative to baseline
                    fft_x, fft_y, _ = run_variant_a(ref_img, search_img)
                    if float(np.hypot(px - fft_x, py - fft_y)) > 5.0:
                        total_overrides += 1
                        car_err = float(np.hypot(px - gt_x, py - gt_y))
                        fft_err = float(np.hypot(fft_x - gt_x, fft_y - gt_y))
                        if car_err <= 5.0 and fft_err > 5.0:
                            beneficial_overrides += 1
                        elif car_err > 5.0 and fft_err <= 5.0:
                            harmful_overrides += 1
            elif var == "Variant_E_CAR_RGB_Bonus":
                px, py, cands, _ = run_variant_e(ref_img, search_img, model, device)

            dt = (time.perf_counter() - t0) * 1000.0
            err = float(np.hypot(px - gt_x, py - gt_y))
            errors.append(err)
            latencies.append(dt)

            # Candidate Recall audit (distance <= 8.0 px)
            ranks = [r for r, c in enumerate(cands, start=1) if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 8.0]
            top1_recalls.append(1 if len(ranks) > 0 and ranks[0] == 1 else 0)
            top5_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 5 else 0)
            top10_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 10 else 0)
            top20_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 20 else 0)

        err_arr = np.array(errors)
        lat_arr = np.array(latencies)

        results[var] = {
            "Top1_Recall": round(float(np.mean(top1_recalls)) * 100, 2),
            "Top5_Recall": round(float(np.mean(top5_recalls)) * 100, 2),
            "Top10_Recall": round(float(np.mean(top10_recalls)) * 100, 2),
            "Top20_Recall": round(float(np.mean(top20_recalls)) * 100, 2),
            "Acc_le1px": round(float(np.mean(err_arr <= 1.0)) * 100, 2),
            "Acc_le3px": round(float(np.mean(err_arr <= 3.0)) * 100, 2),
            "Acc_le5px": round(float(np.mean(err_arr <= 5.0)) * 100, 2),
            "Acc_le10px": round(float(np.mean(err_arr <= 10.0)) * 100, 2),
            "Acc_le25px": round(float(np.mean(err_arr <= 25.0)) * 100, 2),
            "Mean_Err": round(float(np.mean(err_arr)), 2),
            "Median_Err": round(float(np.median(err_arr)), 2),
            "P95_Err": round(float(np.percentile(err_arr, 95)), 2),
            "Mean_Lat": round(float(np.mean(lat_arr)), 2),
            "P95_Lat": round(float(np.percentile(lat_arr, 95)), 2),
            "Max_Lat": round(float(np.max(lat_arr)), 2),
            "pace_activations": pace_activations,
            "total_overrides": total_overrides,
            "beneficial_overrides": beneficial_overrides,
            "harmful_overrides": harmful_overrides,
        }

    # Print Master Summary Table
    print(f"\n{'='*110}")
    print(f"  MASTER DUAL-CHANNEL RETRIEVAL ABLATION RESULTS ({label})")
    print(f"{'='*110}")
    header1 = f"{'Variant':<28s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'Top-20':>7s} {'<=1px':>7s} {'<=3px':>7s} {'<=5px':>7s} {'<=10px':>7s} {'<=25px':>7s}"
    print(header1)
    print("-" * len(header1))

    for var, st in results.items():
        print(f"{var:<28s} {st['Top1_Recall']:>6.1f}% {st['Top5_Recall']:>6.1f}% {st['Top10_Recall']:>6.1f}% {st['Top20_Recall']:>6.1f}% "
              f"{st['Acc_le1px']:>6.1f}% {st['Acc_le3px']:>6.1f}% {st['Acc_le5px']:>6.1f}% {st['Acc_le10px']:>6.1f}% {st['Acc_le25px']:>6.1f}%")

    print(f"\n{'='*110}")
    header2 = f"{'Variant':<28s} {'MeanErr':>8s} {'MedErr':>8s} {'P95Err':>8s} {'MeanLat':>9s} {'P95Lat':>9s} {'MaxLat':>9s}"
    print(header2)
    print("-" * len(header2))
    for var, st in results.items():
        print(f"{var:<28s} {st['Mean_Err']:>8.2f} {st['Median_Err']:>8.2f} {st['P95_Err']:>8.2f} {st['Mean_Lat']:>8.2f}ms {st['P95_Lat']:>8.2f}ms {st['Max_Lat']:>8.2f}ms")

    print(f"\n{'='*110}")
    print("  CAR SAFETY & AI OVERRIDE AUDIT:")
    st_d = results["Variant_D_CAR_Final"]
    print(f"    - PACE Activations: {st_d['pace_activations']}/{n} ({st_d['pace_activations']/n*100:.1f}%)")
    print(f"    - Total FFT Overrides: {st_d['total_overrides']}")
    print(f"    - Beneficial Overrides (FFT Wrong -> CAR Correct): {st_d['beneficial_overrides']}")
    print(f"    - Harmful Overrides (FFT Correct -> CAR Wrong): {st_d['harmful_overrides']}")
    cor_rate = (st_d['beneficial_overrides'] / max(1, st_d['total_overrides'])) * 100 if st_d['total_overrides'] > 0 else 0.0
    print(f"    - Beneficial Override Rate: {cor_rate:.1f}%")
    print(f"{'='*110}")

    return results


def main():
    os.makedirs("results", exist_ok=True)
    res_holdout = evaluate_ablation_on_manifest(HOLDOUT_MANIFEST, "Held-Out Test Set (200 Samples)")

    if res_holdout:
        rows = []
        for var, st in res_holdout.items():
            rows.append({
                "Variant": var,
                "Top1_Recall": st["Top1_Recall"],
                "Top5_Recall": st["Top5_Recall"],
                "Top10_Recall": st["Top10_Recall"],
                "Top20_Recall": st["Top20_Recall"],
                "Acc_le1px": st["Acc_le1px"],
                "Acc_le3px": st["Acc_le3px"],
                "Acc_le5px": st["Acc_le5px"],
                "Acc_le10px": st["Acc_le10px"],
                "Acc_le25px": st["Acc_le25px"],
                "Mean_Err": st["Mean_Err"],
                "Median_Err": st["Median_Err"],
                "P95_Err": st["P95_Err"],
                "Mean_Lat_ms": st["Mean_Lat"],
                "P95_Lat_ms": st["P95_Lat"],
                "Max_Lat_ms": st["Max_Lat"],
            })
        df_out = pd.DataFrame(rows)
        csv_path = "results/dual_channel_ablation_results.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\nSaved master ablation results to '{csv_path}'")


if __name__ == "__main__":
    main()
