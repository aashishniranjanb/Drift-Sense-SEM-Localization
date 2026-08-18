"""
Top-K Retrieval Recall & Diagnostic Failure Taxonomy
Evaluates:
1. Top-K Recall (K = 1, 3, 5, 10, 20) for FFT-NCC vs HCR on the frozen 200-case test set
2. Detailed Failure Breakdown:
   - Retrieval Failure: GT is NOT in Top-20 candidates
   - Ranking Failure: GT is in Top-20, but ranked #2, #3, or lower
   - Subpixel/Tie Failure: GT is ranked #1, but subpixel refinement or center prior shifted it > 5 px away
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd

from inference_hcr import load_siamese_model, siamese_rerank_candidates, extract_patch_safe, normalize_intensity
from hf_space.baseline_solution.zncc import zncc_match

MANIFEST_PATH = "data/hcr_test/manifest.csv"


def extract_top_k_candidates_fft(search_img: np.ndarray, ref_img: np.ndarray, k_max: int = 20) -> list[dict]:
    """Extract Top-K spatial correlation peaks using FFT-NCC with NMS."""
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    c_i = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    work = c_i.copy()
    ch, cw = work.shape
    candidates = []

    for _ in range(k_max):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + 50.0
        cy = py + 50.0
        candidates.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "ncc_score": float(max_val),
            "corr_plane": c_i
        })
        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0

    return candidates


def diagnose_sample(ref_img: np.ndarray, search_img: np.ndarray, gt_x: float, gt_y: float, model, device, correct_radius: float = 8.0) -> dict:
    """Diagnoses a single sample to locate ground truth rank and failure mode."""
    candidates_fft = extract_top_k_candidates_fft(search_img, ref_img, k_max=20)

    # Label candidates by ground truth distance
    gt_ranks_fft = []
    for rank, c in enumerate(candidates_fft, start=1):
        err = float(np.hypot(c["cx"] - gt_x, c["cy"] - gt_y))
        c["err_to_gt"] = err
        c["fft_rank"] = rank
        if err <= correct_radius:
            gt_ranks_fft.append(rank)

    gt_in_fft_top1 = len(gt_ranks_fft) > 0 and gt_ranks_fft[0] == 1
    gt_in_fft_top3 = len(gt_ranks_fft) > 0 and gt_ranks_fft[0] <= 3
    gt_in_fft_top5 = len(gt_ranks_fft) > 0 and gt_ranks_fft[0] <= 5
    gt_in_fft_top10 = len(gt_ranks_fft) > 0 and gt_ranks_fft[0] <= 10
    gt_in_fft_top20 = len(gt_ranks_fft) > 0 and gt_ranks_fft[0] <= 20

    # Evaluate Siamese re-ranking if model is available
    gt_in_hcr_top1 = False
    gt_ranks_hcr = []
    if model is not None and len(candidates_fft) > 0:
        cands_hcr = [dict(c) for c in candidates_fft]
        cands_hcr = siamese_rerank_candidates(model, device, ref_img, search_img, cands_hcr)
        for c in cands_hcr:
            c["hcr_score"] = 0.60 * c["ncc_score"] + 0.40 * max(0.0, c.get("neural_sim", 0.0))
        cands_hcr.sort(key=lambda c: c["hcr_score"], reverse=True)

        for rank, c in enumerate(cands_hcr, start=1):
            if c["err_to_gt"] <= correct_radius:
                gt_ranks_hcr.append(rank)

        gt_in_hcr_top1 = len(gt_ranks_hcr) > 0 and gt_ranks_hcr[0] == 1

    # Failure Taxonomy
    failure_type = "SUCCESS"
    if not gt_in_fft_top20:
        failure_type = "RETRIEVAL_FAILURE"  # True site not present in Top-20
    elif not gt_in_fft_top1:
        failure_type = "RANKING_FAILURE"    # True site in Top-20, but FFT ranked periodic replica higher
    elif gt_in_fft_top1 and not gt_in_hcr_top1 and model is not None:
        failure_type = "SIAMESE_DEGRADATION" # FFT had it at #1, but Siamese demoted it

    return {
        "gt_in_fft_top1": gt_in_fft_top1,
        "gt_in_fft_top3": gt_in_fft_top3,
        "gt_in_fft_top5": gt_in_fft_top5,
        "gt_in_fft_top10": gt_in_fft_top10,
        "gt_in_fft_top20": gt_in_fft_top20,
        "gt_rank_fft": gt_ranks_fft[0] if len(gt_ranks_fft) > 0 else 999,
        "gt_in_hcr_top1": gt_in_hcr_top1,
        "gt_rank_hcr": gt_ranks_hcr[0] if len(gt_ranks_hcr) > 0 else 999,
        "failure_type": failure_type,
        "top1_fft_score": candidates_fft[0]["ncc_score"] if len(candidates_fft) > 0 else 0.0,
        "top2_fft_score": candidates_fft[1]["ncc_score"] if len(candidates_fft) > 1 else 0.0,
        "delta_s": (candidates_fft[0]["ncc_score"] - candidates_fft[1]["ncc_score"]) if len(candidates_fft) > 1 else 0.0,
    }


def main():
    print("=" * 90)
    print("      TOP-K RETRIEVAL RECALL & FAILURE TAXONOMY DIAGNOSTIC (200 SAMPLES)      ")
    print("=" * 90)

    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest '{MANIFEST_PATH}' not found!")
        return

    df = pd.read_csv(MANIFEST_PATH)
    model, device = load_siamese_model()

    results = []
    diff_stats = {d: {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0}
                  for d in ["Easy", "Medium", "Hard", "Adversarial"]}
    failure_counts = {"SUCCESS": 0, "RETRIEVAL_FAILURE": 0, "RANKING_FAILURE": 0, "SIAMESE_DEGRADATION": 0}

    for idx, row in df.iterrows():
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        diff = row.get("difficulty", "Unknown")

        res = diagnose_sample(ref_img, search_img, gt_x, gt_y, model, device)
        results.append(res)

        if diff in diff_stats:
            diff_stats[diff]["total"] += 1
            if res["gt_in_fft_top1"]: diff_stats[diff]["top1"] += 1
            if res["gt_in_fft_top3"]: diff_stats[diff]["top3"] += 1
            if res["gt_in_fft_top5"]: diff_stats[diff]["top5"] += 1
            if res["gt_in_fft_top10"]: diff_stats[diff]["top10"] += 1
            if res["gt_in_fft_top20"]: diff_stats[diff]["top20"] += 1

        failure_counts[res["failure_type"]] += 1

    df_res = pd.DataFrame(results)
    total = len(df_res)

    print("\nOVERALL TOP-K CANDIDATE RETRIEVAL RECALL:")
    print(f"  - Top-1  Recall: {df_res['gt_in_fft_top1'].mean()*100:5.2f}% ({df_res['gt_in_fft_top1'].sum()}/{total})")
    print(f"  - Top-3  Recall: {df_res['gt_in_fft_top3'].mean()*100:5.2f}% ({df_res['gt_in_fft_top3'].sum()}/{total})")
    print(f"  - Top-5  Recall: {df_res['gt_in_fft_top5'].mean()*100:5.2f}% ({df_res['gt_in_fft_top5'].sum()}/{total})")
    print(f"  - Top-10 Recall: {df_res['gt_in_fft_top10'].mean()*100:5.2f}% ({df_res['gt_in_fft_top10'].sum()}/{total})")
    print(f"  - Top-20 Recall: {df_res['gt_in_fft_top20'].mean()*100:5.2f}% ({df_res['gt_in_fft_top20'].sum()}/{total})")

    print("\nTOP-K RECALL BREAKDOWN BY DIFFICULTY:")
    for d, st in diff_stats.items():
        if st["total"] > 0:
            print(f"  [{d:<11s}]: Top-1={st['top1']/st['total']*100:4.1f}% | Top-3={st['top3']/st['total']*100:4.1f}% | "
                  f"Top-5={st['top5']/st['total']*100:4.1f}% | Top-10={st['top10']/st['total']*100:4.1f}% | Top-20={st['top20']/st['total']*100:4.1f}%")

    print("\nFAILURE TAXONOMY BREAKDOWN:")
    for ftype, count in failure_counts.items():
        pct = (count / total) * 100
        print(f"  - [{ftype:<20s}]: {count:3d} samples ({pct:5.1f}%)")

    print("=" * 90)


if __name__ == "__main__":
    main()
