import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy import stats

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "phase2"))
sys.path.append(os.path.join(parent_dir, "fallbacks"))
sys.path.append(os.path.join(parent_dir, "production_engine"))
sys.path.append(os.path.join(parent_dir, "team", "akhilesh-localization"))

from fallbacks.pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from family_clustering import cluster_replica_families
from spatial_fingerprint import compute_spatial_fingerprint
from context_matcher import verify_candidate_context
from inference_phase2 import (
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation
)

def run_v17_replica_35_audit():
    os.makedirs("results/v17", exist_ok=True)
    
    # 1. Load V16 failure taxonomy to get the 35 PERIODIC_REPLICA cases
    tax_path = "results/v16/failure_taxonomy.csv"
    if not os.path.exists(tax_path):
        print(f"Error: {tax_path} not found.")
        return
        
    tax_df = pd.read_csv(tax_path)
    replica_failures = tax_df[tax_df["failure_mode"] == "PERIODIC_REPLICA"].copy()
    print(f"Loaded {len(replica_failures)} PERIODIC_REPLICA failure cases from V16.")
    
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    
    records = []
    
    for idx, fail_row in replica_failures.iterrows():
        pair_id = fail_row["pair_id"]
        pair_data = pairs_df[pairs_df["pair_id"] == pair_id].iloc[0]
        
        gt_x = float(pair_data["gt_x"])
        gt_y = float(pair_data["gt_y"])
        gt_scale = float(pair_data.get("gt_scale", 10.0))
        gt_theta = float(pair_data.get("gt_theta", 0.0))
        set_type = pair_data.get("set_type", "SetA")
        
        ref_path = os.path.join("data/phase2_dev", pair_data["reference_path"])
        search_path = os.path.join("data/phase2_dev", pair_data["search_path"])
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        sh, sw = search_img.shape[:2]
        search_cx, search_cy = sw / 2.0, sh / 2.0
        
        # 1. Pose estimation (Sequential Fallback)
        pose_res = perform_pose_fallback_search(ref_img, search_img)
        est_scale = float(pose_res["best_scale"])
        est_theta = float(pose_res["best_theta"])
        corr_plane = pose_res["corr_plane"]
        rotated_template = pose_res["best_template"]
        th, tw = rotated_template.shape[:2]
        
        # 2. Candidate extraction (V16 Akhilesh Rescue Queue)
        candidates = extract_candidates_akhilesh(
            corr_plane, tw, th, ref_img=ref_img, search_img=search_img,
            est_scale=est_scale, est_theta=est_theta
        )
        
        # 3. Enrich features for all candidates
        candidates = cluster_replica_families(candidates, est_scale)
        
        enriched = []
        for c in candidates:
            px, py = c["peak_x"], c["peak_y"]
            cx, cy = c["cx"], c["cy"]
            
            y1, y2 = max(0, int(py)), min(sh, int(py + th))
            x1, x2 = max(0, int(px)), min(sw, int(px + tw))
            search_crop = search_img[y1:y2, x1:x2]
            
            # PSR
            psr, _, _ = compute_psr(corr_plane, px, py)
            
            # Context
            ctx_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
            
            # Phase correlation & residual
            phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
            if search_crop.shape == (th, tw):
                phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_template, search_crop)
                
            # SSD
            ssd = 0.0
            if search_crop.shape == (th, tw):
                ssd = float(np.mean((search_crop.astype(np.float32) - rotated_template.astype(np.float32)) ** 2))
                
            # Phase penalty
            phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
            
            # Center distance (Search center prior)
            dist_to_center = float(np.hypot(cx - search_cx, cy - search_cy))
            
            # Spatial fingerprint
            fam_members = [m for m in candidates if m.get("family_id") == c.get("family_id")]
            fp = compute_spatial_fingerprint(search_img, cx, cy, est_scale, fam_members)
            
            c_dict = {
                "peak_x": px,
                "peak_y": py,
                "cx": cx,
                "cy": cy,
                "corr_score": c.get("corr_score", 0.0),
                "rescue_score": c.get("rescue_score", 0.0),
                "psr": psr,
                "phase_residual": phase_residual,
                "phase_penalty": phase_penalty,
                "context_32": ctx_res.get("s32", 0.0),
                "context_64": ctx_res.get("s64", 0.0),
                "context_128": ctx_res.get("s128", 0.0),
                "context_combined": ctx_res.get("combined", 0.0),
                "ssd": ssd,
                "dist_to_center": dist_to_center,
                "nearest_edge_dist": fp.get("nearest_edge_dist", 0.0),
                "nearest_cut_dist": fp.get("nearest_cut_dist", 0.0),
                "row_spacing": fp.get("row_spacing", 0.0),
                "col_spacing": fp.get("col_spacing", 0.0),
                "local_density": fp.get("local_density", 0.0),
                "family_id": c.get("family_id", -1),
                "family_population": c.get("family_population", 1),
                "family_score_variance": c.get("family_score_variance", 0.0)
            }
            enriched.append(c_dict)
            
        # Current winner is candidate #0 (or rank 1 from current ranking)
        win_cand = enriched[0] if len(enriched) > 0 else None
        
        # Locate Ground Truth candidate (<= 5 px)
        gt_cand = None
        gt_rank_in_pool = -1
        for r_idx, c in enumerate(enriched):
            d = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
            if d <= 5.0:
                gt_cand = c
                gt_rank_in_pool = r_idx + 1
                break
                
        rec = {
            "pair_id": pair_id,
            "set_type": set_type,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_present_in_top50": 1 if gt_cand is not None else 0,
            "gt_rank_in_pool": gt_rank_in_pool,
            "win_x": win_cand["cx"] if win_cand else 0.0,
            "win_y": win_cand["cy"] if win_cand else 0.0,
            "win_error": float(np.hypot(win_cand["cx"] - gt_x, win_cand["cy"] - gt_y)) if win_cand else 999.0,
        }
        
        feature_names = [
            "corr_score", "psr", "phase_residual", "phase_penalty",
            "context_32", "context_64", "context_128", "context_combined",
            "ssd", "dist_to_center", "nearest_edge_dist", "nearest_cut_dist",
            "row_spacing", "col_spacing", "local_density", "family_population"
        ]
        
        for f in feature_names:
            rec[f"gt_{f}"] = gt_cand[f] if gt_cand else np.nan
            rec[f"win_{f}"] = win_cand[f] if win_cand else np.nan
            if gt_cand and win_cand:
                rec[f"diff_{f}"] = gt_cand[f] - win_cand[f]
            else:
                rec[f"diff_{f}"] = np.nan
                
        records.append(rec)
        print(f"Audited {pair_id} ({set_type}): GT in pool = {rec['gt_present_in_top50']} (Rank {gt_rank_in_pool}), Win Err = {rec['win_error']:.1f} px")
        
    audit_df = pd.DataFrame(records)
    csv_out = "results/v17/V17_REPLICA_35_AUDIT.csv"
    audit_df.to_csv(csv_out, index=False)
    print(f"\nSaved audit CSV to {csv_out}")
    
    # 4. Generate In-Depth Forensic Statistical Report
    # Analyze the subset where GT is actually present in Top-50 pool
    gt_in_pool_df = audit_df[audit_df["gt_present_in_top50"] == 1].copy()
    num_gt_in_pool = len(gt_in_pool_df)
    total_failures = len(audit_df)
    
    report_lines = []
    report_lines.append("# V17 Replica Discrimination Forensic Report (35 Periodic Failure Audit)\n")
    report_lines.append(f"**Total Periodic-Replica Failures Audited:** {total_failures}\n")
    report_lines.append(f"**Failures with GT Present in Top-50 Pool:** {num_gt_in_pool} / {total_failures} ({num_gt_in_pool/total_failures*100:.1f}%)\n")
    report_lines.append(f"**Failures with GT Missing from Top-50 Pool (Retrieval Caps):** {total_failures - num_gt_in_pool} ({ (total_failures - num_gt_in_pool)/total_failures*100:.1f}%)\n\n")
    report_lines.append("## Statistical Feature Separability (GT vs Winning False Replica)\n")
    report_lines.append("Analyzing cases where the True GT candidate was inside the Top-50 pool but lost to a false replica at Rank #1:\n\n")
    report_lines.append("| Feature | GT Mean | Winner Mean | Mean Diff (GT - Win) | GT Win-Rate (%) | T-Stat | P-Value | Direction |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    stat_summary = []
    for f in feature_names:
        gt_vals = gt_in_pool_df[f"gt_{f}"].dropna()
        win_vals = gt_in_pool_df[f"win_{f}"].dropna()
        diff_vals = gt_in_pool_df[f"diff_{f}"].dropna()
        
        if len(diff_vals) > 2:
            gt_mean = gt_vals.mean()
            win_mean = win_vals.mean()
            mean_diff = diff_vals.mean()
            
            # Direction: Higher is better or Lower is better?
            # Win rate: For features where higher = GT, count diff > 0. For features where lower = GT, count diff < 0.
            gt_greater_pct = (diff_vals > 0).mean() * 100
            
            # Paired T-test
            t_stat, p_val = stats.ttest_rel(gt_vals, win_vals)
            
            # Best interpretation
            if mean_diff > 0:
                direction = "GT Higher"
                effective_win_rate = gt_greater_pct
            else:
                direction = "GT Lower"
                effective_win_rate = 100.0 - gt_greater_pct
                
            stat_summary.append({
                "feature": f,
                "gt_mean": gt_mean,
                "win_mean": win_mean,
                "mean_diff": mean_diff,
                "win_rate": effective_win_rate,
                "t_stat": t_stat,
                "p_val": p_val,
                "direction": direction
            })
            
            report_lines.append(
                f"| `{f}` | {gt_mean:.4f} | {win_mean:.4f} | {mean_diff:+.4f} | {effective_win_rate:.1f}% | {t_stat:.2f} | {p_val:.4f} | {direction} |"
            )
            
    report_lines.append("\n## Key Forensic Findings & Conclusions\n")
    
    # Sort features by absolute t_stat or p_value to highlight the strongest discriminators
    stat_summary_sorted = sorted(stat_summary, key=lambda x: abs(x["t_stat"]), reverse=True)
    report_lines.append("### Top 5 Strongest Replica Discriminators:\n")
    for rank, s in enumerate(stat_summary_sorted[:5], 1):
        report_lines.append(f"{rank}. **`{s['feature']}`**: Mean Diff = {s['mean_diff']:+.4f}, Win-Rate = {s['win_rate']:.1f}%, p = {s['p_val']:.4e} ({s['direction']})")
        
    report_path = "results/v17/V17_REPLICA_35_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    print(f"Saved forensic report to {report_path}")

if __name__ == "__main__":
    run_v17_replica_35_audit()
