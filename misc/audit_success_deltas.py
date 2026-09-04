import sys, os, pickle, cv2
sys.path.insert(0, 'FINAL_SUBMISSION/runtime/src')
import numpy as np, pandas as pd
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
from periodicity_detector import estimate_periodicity_from_corr

def main():
    pool_audit = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    raw_v25 = pd.read_csv("data/phase2_dev/v25_predictions.csv")

    success_pids = pool_audit[pool_audit["category"] == "SUCCESS_ACCEPTED"]["pair_id"].tolist()
    print(f"Extracting features for top 5 candidates of 76 successful pairs...")

    records = []
    # Sample 20 successes for fast profile comparison
    sample_success = success_pids[:20]

    for pid in sample_success:
        row = gt_df[gt_df["pair_id"] == pid].iloc[0]
        v25_row = raw_v25[raw_v25["pair_id"] == pid].iloc[0]
        gt_x, gt_y = row["gt_x"], row["gt_y"]
        est_scale = float(v25_row["scale"])
        est_theta = float(v25_row["theta"])

        ref = cv2.imread(os.path.join("data/phase2_dev", row["reference_path"]), cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join("data/phase2_dev", row["search_path"]), cv2.IMREAD_GRAYSCALE)
        tw, th = int(round(ref.shape[1] / est_scale)), int(round(ref.shape[0] / est_scale))
        tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        if abs(est_theta) > 0.01:
            tpl_rot = rotate_image(tpl, est_theta)
        else:
            tpl_rot = tpl
        corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)
        cands = extract_candidates_akhilesh(corr_plane, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)

        # Candidate 0 is Top-1 (GT)
        # Candidate 1 is Top-2 (Best replica)
        top1 = cands[0]
        top2 = cands[1] if len(cands) > 1 else top1

        ctx1 = verify_candidate_context(ref, srch, top1["cx"], top1["cy"], est_scale, est_theta)
        ctx2 = verify_candidate_context(ref, srch, top2["cx"], top2["cy"], est_scale, est_theta)

        per = estimate_periodicity_from_corr(corr_plane)
        neigh1 = compute_neighborhood_consistency(srch, tpl_rot, top1["peak_x"], top1["peak_y"], per["pitch_x"], per["pitch_y"])
        neigh2 = compute_neighborhood_consistency(srch, tpl_rot, top2["peak_x"], top2["peak_y"], per["pitch_x"], per["pitch_y"])

        records.append({
            "pair_id": pid,
            "d_corr (top1 - top2)": top1["corr_score"] - top2["corr_score"],
            "d_ctx (top1 - top2)": ctx1["combined"] - ctx2["combined"],
            "d_neigh (top1 - top2)": neigh1 - neigh2
        })

    df = pd.DataFrame(records)
    print("\nSUCCESS CASES (Top1=GT vs Top2=Replica):")
    print(f"  Mean d_corr : {df['d_corr (top1 - top2)'].mean():+.4f}")
    print(f"  Mean d_ctx  : {df['d_ctx (top1 - top2)'].mean():+.4f}")
    print(f"  Mean d_neigh: {df['d_neigh (top1 - top2)'].mean():+.4f}")

if __name__ == "__main__":
    main()
