import os
import sys
import time
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh

def main():
    print("==================================================================", flush=True)
    print("   COMPLETE 180-PAIR FAILURE DECOMPOSITION & CEILING AUDIT        ", flush=True)
    print("==================================================================", flush=True)

    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    pred_df = pd.read_csv("FINAL_SUBMISSION/validation/scale_only.csv")
    audit_df = pd.read_csv("data/phase2_dev/score_audit_180.csv")
    cache_df = pd.read_csv("FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv")
    raw_v25 = pd.read_csv("data/phase2_dev/v25_predictions.csv")

    dev_df = pd.merge(gt_df, pred_df, on="pair_id", suffixes=("_gt", "_pred"))
    dev_df = pd.merge(dev_df, cache_df, on="pair_id", suffixes=("", "_cache"))
    dev_df = pd.merge(dev_df, audit_df[["pair_id", "localization_error", "localization_tier"]], on="pair_id")
    dev_df = pd.merge(dev_df, raw_v25[["pair_id", "theta", "scale"]], on="pair_id", suffixes=("", "_raw"))

    results = []
    t0 = time.time()

    present_df = dev_df[dev_df["gt_found"] == 1].copy()
    print(f"Auditing all {len(present_df)} PRESENT pairs with exact V25 pose parameters...", flush=True)

    for i, (_, row) in enumerate(present_df.iterrows()):
        pid = row["pair_id"]
        set_type = row["set_type"]
        gt_x, gt_y = row["gt_x"], row["gt_y"]
        pred_found = row["found"]
        top1_err = row["localization_error"]
        est_scale = float(row["scale_raw"])
        est_theta = float(row["theta_raw"])

        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))

        ref_img = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        srch_img = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)

        ref_f = ref_img.astype(np.float32)
        srch_f = srch_img.astype(np.float32)
        ref_h, ref_w = ref_f.shape[:2]

        tw = int(round(ref_w / est_scale))
        th = int(round(ref_h / est_scale))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)

        if abs(est_theta) > 0.01:
            tpl_rot = rotate_image(tpl, est_theta)
        else:
            tpl_rot = tpl

        corr_plane = cv2.matchTemplate(srch_f, tpl_rot, cv2.TM_CCOEFF_NORMED)
        cands = extract_candidates_akhilesh(corr_plane, tw, th, ref_img, srch_img, est_scale, est_theta, max_final_k=200)

        # Compute error for all candidates in the pool
        cand_errors = [float(np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)) for c in cands]
        min_pool_err = min(cand_errors) if cand_errors else 999.0
        n_le5 = sum(e <= 5.0 for e in cand_errors)
        n_le10 = sum(e <= 10.0 for e in cand_errors)
        n_le25 = sum(e <= 25.0 for e in cand_errors)

        # Classification
        if pred_found == 1:
            if top1_err <= 5.0:
                cat = "SUCCESS_ACCEPTED"
            else:
                cat = "FALSE_ACCEPT_WRONG_LOC"
        else: # pred_found == 0
            if top1_err <= 5.0:
                cat = "REJECTION_FAILURE" # Type 1: Top-1 was right, but rejected
            elif min_pool_err <= 5.0:
                cat = "RANKING_FAILURE"   # Type 2: Top-1 was wrong (>5px), but candidate pool contains <=5px match
            else:
                cat = "RETRIEVAL_FAILURE" # Type 3: Pool contains NO candidate <= 5px

        results.append({
            "pair_id": pid,
            "set_type": set_type,
            "pred_found": pred_found,
            "top1_err": top1_err,
            "min_pool_err": min_pool_err,
            "cands_le5": n_le5,
            "cands_le10": n_le10,
            "cands_le25": n_le25,
            "category": cat
        })

        if (i + 1) % 35 == 0 or (i + 1) == len(present_df):
            print(f"[{i+1}/{len(present_df)}] elapsed: {time.time()-t0:.1f}s", flush=True)

    res_df = pd.DataFrame(results)

    # Summary of Failure Categories
    print("\n" + "="*60, flush=True)
    print("       FAILURE DECOMPOSITION SUMMARY (140 PRESENT PAIRS)      ", flush=True)
    print("="*60, flush=True)
    counts = res_df["category"].value_counts()
    for cat, count in counts.items():
        pct = (count / len(present_df)) * 100.0
        print(f"  {cat:<25s}: {count:3d} / 140 ({pct:5.1f}%)", flush=True)

    # Breakdown by SetA vs SetB
    print("\nBreakdown by Dataset Split:", flush=True)
    print(pd.crosstab(res_df["set_type"], res_df["category"]), flush=True)

    # Analyze the Ranking Failures
    ranking_fails = res_df[res_df["category"] == "RANKING_FAILURE"]
    print(f"\n[RANKING FAILURES (Type 2)]: n={len(ranking_fails)}", flush=True)
    print(f"Average min error in pool: {ranking_fails['min_pool_err'].mean():.2f}px", flush=True)
    print(f"Distribution of min pool error:", flush=True)
    print(f"  <= 1.0 px: {sum(ranking_fails['min_pool_err'] <= 1.0)}", flush=True)
    print(f"  <= 2.0 px: {sum(ranking_fails['min_pool_err'] <= 2.0)}", flush=True)
    print(f"  <= 3.0 px: {sum(ranking_fails['min_pool_err'] <= 3.0)}", flush=True)
    print(f"  <= 5.0 px: {sum(ranking_fails['min_pool_err'] <= 5.0)}", flush=True)

    # Analyze Retrieval Failures
    retrieval_fails = res_df[res_df["category"] == "RETRIEVAL_FAILURE"]
    print(f"\n[RETRIEVAL FAILURES (Type 3)]: n={len(retrieval_fails)}", flush=True)
    print(f"Average min error in pool: {retrieval_fails['min_pool_err'].mean():.2f}px", flush=True)
    print(f"  <= 10 px: {sum(retrieval_fails['min_pool_err'] <= 10.0)}", flush=True)
    print(f"  <= 25 px: {sum(retrieval_fails['min_pool_err'] <= 25.0)}", flush=True)
    print(f"  > 25 px : {sum(retrieval_fails['min_pool_err'] > 25.0)}", flush=True)

    # CEILINGS CALCULATION:
    n_rank_recoverable = len(ranking_fails)
    total_loc_recoverable = 76 + 2 + n_rank_recoverable

    print("\n" + "="*60, flush=True)
    print("             COMPREHENSIVE CEILINGS ANALYSIS                 ", flush=True)
    print("="*60, flush=True)
    print(f"1. CURRENT BASELINE (Frozen Engine):", flush=True)
    print(f"   Localized <= 5px: 76 / 140 present (SetA: 40/70, SetB: 36/70)", flush=True)
    print(f"   Official Loc Score: 40.00 / 40.00 (100% of accepted)", flush=True)
    print(f"   Official Rej Score: 8.03 / 15.00", flush=True)

    print(f"\n2. PURE REJECTION CEILING (Frozen Top-1 Ranker):", flush=True)
    print(f"   Max Recoverable: 2 pairs (pair_027, pair_078)", flush=True)
    print(f"   Max Loc: 40.00 / 40.00", flush=True)
    print(f"   Max Rej F1: 0.5634 -> 8.45 / 15.00 (+0.42 points)", flush=True)
    print(f"   Max Total Score: ~90.98", flush=True)

    fp_rank = 140 - total_loc_recoverable
    prec_rank = 40.0 / (40.0 + fp_rank)
    rec_rank = 1.0
    f1_rank = 2 * prec_rank * rec_rank / (prec_rank + rec_rank)
    rej_pts_rank = f1_rank * 15.0

    print(f"\n3. RE-RANKING CEILING (Candidate Pool Top-200 Re-Ranker):", flush=True)
    print(f"   Max Localized <= 5px: {total_loc_recoverable} / 140 ({total_loc_recoverable/140*100:.1f}%)", flush=True)
    print(f"   Remaining FP (pure retrieval failures): {fp_rank}", flush=True)
    print(f"   Max Rej F1: {f1_rank:.4f} -> {rej_pts_rank:.2f} / 15.00 (+{rej_pts_rank - 8.03:.2f} points)", flush=True)
    print(f"   Loc Score: 40.00 / 40.00 (all accepted candidates <= 5px)", flush=True)
    print(f"   Max Total Score: ~{40.0 + 19.74 + rej_pts_rank + 8.27 + 5.0 + 10.0:.2f} / 100.00", flush=True)

    res_df.to_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv", index=False)
    print(f"\nSaved detailed per-pair analysis to 'FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv'.", flush=True)

if __name__ == "__main__":
    main()
