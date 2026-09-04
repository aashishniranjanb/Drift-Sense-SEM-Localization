import os
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
from periodicity_detector import estimate_periodicity_from_corr

def main():
    print("==================================================================")
    print("      DEEP FORENSIC AUDIT OF THE 26 RANKING FAILURE CASES         ")
    print("==================================================================")

    # 1. Load data
    pool_audit = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    raw_v25 = pd.read_csv("data/phase2_dev/v25_predictions.csv")

    ranking_fails = pool_audit[pool_audit["category"] == "RANKING_FAILURE"]["pair_id"].tolist()
    print(f"Auditing all {len(ranking_fails)} ranking failure pairs...\n")

    records = []

    for pid in ranking_fails:
        row = gt_df[gt_df["pair_id"] == pid].iloc[0]
        v25_row = raw_v25[raw_v25["pair_id"] == pid].iloc[0]

        gt_x, gt_y = row["gt_x"], row["gt_y"]
        set_type = row["set_type"]
        est_scale = float(v25_row["scale"])
        est_theta = float(v25_row["theta"])

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
        cands = cluster_replica_families(cands, est_scale)

        per = estimate_periodicity_from_corr(corr_plane)
        pitch_x, pitch_y = per["pitch_x"], per["pitch_y"]

        # Extract features for all candidates
        for c in cands:
            cx, cy = c["cx"], c["cy"]
            px, py = c["peak_x"], c["peak_y"]
            ctx = verify_candidate_context(ref_img, srch_img, cx, cy, est_scale, est_theta)
            phase_pen = verify_phase_consistency(srch_img, tpl_rot, px, py)
            neigh = compute_neighborhood_consistency(srch_img, tpl_rot, px, py, pitch_x, pitch_y)
            gncc = compute_gradient_ncc(srch_img, tpl_rot, px, py)

            c["context_128"] = ctx["s128"]
            c["context_combined"] = ctx["combined"]
            c["phase_penalty"] = phase_pen
            c["neigh_cons"] = neigh
            c["grad_ncc"] = gncc
            c["err_to_gt"] = float(np.hypot(cx - gt_x, cy - gt_y))

        # Rank 0 candidate in extraction (pre-ranker)
        # Note: In pipeline.py, ranker predicts on candidates, then sorts by ml_score
        # Let's find:
        # 1. Which candidate is the GT candidate (err <= 5px)?
        # 2. Which candidate is Rank 1 (currently winning periodic replica)?
        gt_cands = [c for c in cands if c["err_to_gt"] <= 5.0]
        gt_cands.sort(key=lambda c: c["err_to_gt"])
        best_gt_cand = gt_cands[0] if gt_cands else None

        # The extracted rank of best_gt_cand
        gt_extract_rank = cands.index(best_gt_cand) if best_gt_cand else -1

        top1_replica = cands[0] # extracted top-1 peak

        dist_between = float(np.hypot(best_gt_cand["cx"] - top1_replica["cx"], best_gt_cand["cy"] - top1_replica["cy"])) if best_gt_cand else 0.0

        records.append({
            "pair_id": pid,
            "set_type": set_type,
            "gt_extract_rank": gt_extract_rank,
            "gt_err": best_gt_cand["err_to_gt"] if best_gt_cand else -1,
            "top1_err": top1_replica["err_to_gt"],
            "dist_gt_to_top1": dist_between,
            "pitch_x": pitch_x,
            "pitch_y": pitch_y,
            "corr_gt": best_gt_cand["corr_score"] if best_gt_cand else 0,
            "corr_top1": top1_replica["corr_score"],
            "d_corr (gt - top1)": best_gt_cand["corr_score"] - top1_replica["corr_score"] if best_gt_cand else 0,
            "ctx_gt": best_gt_cand["context_combined"] if best_gt_cand else 0,
            "ctx_top1": top1_replica["context_combined"],
            "d_ctx (gt - top1)": best_gt_cand["context_combined"] - top1_replica["context_combined"] if best_gt_cand else 0,
            "neigh_gt": best_gt_cand["neigh_cons"] if best_gt_cand else 0,
            "neigh_top1": top1_replica["neigh_cons"],
            "d_neigh (gt - top1)": best_gt_cand["neigh_cons"] - top1_replica["neigh_cons"] if best_gt_cand else 0,
            "grad_gt": best_gt_cand["grad_ncc"] if best_gt_cand else 0,
            "grad_top1": top1_replica["grad_ncc"],
            "d_grad (gt - top1)": best_gt_cand["grad_ncc"] - top1_replica["grad_ncc"] if best_gt_cand else 0,
            "phase_gt": best_gt_cand["phase_penalty"] if best_gt_cand else 0,
            "phase_top1": top1_replica["phase_penalty"],
        })

    df = pd.DataFrame(records)

    print("=== SUMMARY OF 26 RANKING FAILURES ===")
    print(f"Total pairs audited: {len(df)}")
    print(f"\nGT Candidate Extraction Rank (in top-200 pool):")
    print(f"  Rank 1-5  : {sum(df['gt_extract_rank'] <= 5)}")
    print(f"  Rank 6-15 : {sum((df['gt_extract_rank'] > 5) & (df['gt_extract_rank'] <= 15))}")
    print(f"  Rank 16-50: {sum((df['gt_extract_rank'] > 15) & (df['gt_extract_rank'] <= 50))}")
    print(f"  Rank > 50 : {sum(df['gt_extract_rank'] > 50)}")
    print(f"  Median GT rank: {df['gt_extract_rank'].median():.1f}, Min={df['gt_extract_rank'].min()}, Max={df['gt_extract_rank'].max()}")

    print(f"\nAverage Feature Deltas (GT candidate - Top-1 Replica):")
    print(f"  d_corr  (NCC difference)        : mean={df['d_corr (gt - top1)'].mean():+.4f} (replica is ahead in raw corr)")
    print(f"  d_ctx   (Context difference)    : mean={df['d_ctx (gt - top1)'].mean():+.4f} (GT is ahead in context!)")
    print(f"  d_neigh (Neighborhood diff)     : mean={df['d_neigh (gt - top1)'].mean():+.4f} (GT is ahead in neighborhood!)")
    print(f"  d_grad  (Gradient NCC diff)     : mean={df['d_grad (gt - top1)'].mean():+.4f} (GT is ahead in gradient!)")

    print(f"\nSignal Comparison (% of pairs where GT candidate beats Top-1 Replica):")
    print(f"  Context beats Replica    : {sum(df['d_ctx (gt - top1)'] > 0)} / 26 ({sum(df['d_ctx (gt - top1)'] > 0)/26*100:.1f}%)")
    print(f"  Neighborhood beats Repl  : {sum(df['d_neigh (gt - top1)'] > 0)} / 26 ({sum(df['d_neigh (gt - top1)'] > 0)/26*100:.1f}%)")
    print(f"  Gradient beats Replica   : {sum(df['d_grad (gt - top1)'] > 0)} / 26 ({sum(df['d_grad (gt - top1)'] > 0)/26*100:.1f}%)")
    print(f"  (Context OR Neigh OR Grad) beats: {sum((df['d_ctx (gt - top1)'] > 0) | (df['d_neigh (gt - top1)'] > 0) | (df['d_grad (gt - top1)'] > 0))} / 26 ({sum((df['d_ctx (gt - top1)'] > 0) | (df['d_neigh (gt - top1)'] > 0) | (df['d_grad (gt - top1)'] > 0))/26*100:.1f}%)")

    print(f"\nDistance to Winning Replica (Periodic Lattice Hop):")
    print(f"  Mean distance: {df['dist_gt_to_top1'].mean():.1f} px")
    print(f"  Median distance: {df['dist_gt_to_top1'].median():.1f} px")
    print(f"  Lattice pitch X: median={df['pitch_x'].median():.1f} px, Y: median={df['pitch_y'].median():.1f} px")

    df.to_csv("FINAL_SUBMISSION/validation/ranking_failures_26_forensics.csv", index=False)
    print("\nSaved detailed 26-pair forensic table to 'FINAL_SUBMISSION/validation/ranking_failures_26_forensics.csv'.")

if __name__ == "__main__":
    main()
