import pandas as pd
import numpy as np

def analyze_gt_vs_replica():
    evidence_df = pd.read_csv("results/phase2/candidate_evidence.csv")
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    
    # Merge GT found status
    df = evidence_df.merge(pairs_df[["pair_id", "gt_found", "gt_x", "gt_y", "set_type"]], on="pair_id")
    
    # Filter present pairs only
    present_pairs = df[df["gt_found"] == 1]["pair_id"].unique()
    
    features = [
        "corr_score", "fft_gradient_score", "psr", "peak_margin",
        "context_32", "context_64", "context_128",
        "phase_residual", "template_residual", "center_prior", "score_combined"
    ]
    
    gt_vs_replica_diffs = []
    gt_vs_top_diffs = []
    
    success_gt_found_count = 0
    total_present_count = len(present_pairs)
    
    for pair_id in present_pairs:
        pair_cands = df[df["pair_id"] == pair_id].sort_values(by="candidate_id")
        
        gt_cands = pair_cands[pair_cands["correct"] == 1]
        replica_cands = pair_cands[pair_cands["correct"] == 0]
        
        if len(gt_cands) == 0 or len(replica_cands) == 0:
            continue
            
        success_gt_found_count += 1
        gt_cand = gt_cands.iloc[0]
        top_cand = pair_cands.iloc[0]
        
        # Find nearest replica to GT in spatial coordinates
        replica_cands = replica_cands.copy()
        replica_cands["dist_to_gt"] = np.hypot(replica_cands["cx"] - gt_cand["cx"], replica_cands["cy"] - gt_cand["cy"])
        nearest_replica = replica_cands.sort_values(by="dist_to_gt").iloc[0]
        
        diff_dict = {"pair_id": pair_id, "set_type": gt_cand["set_type"], "dist_to_replica": nearest_replica["dist_to_gt"]}
        for f in features:
            diff_dict[f"delta_{f}"] = gt_cand[f] - nearest_replica[f]
            diff_dict[f"gt_{f}"] = gt_cand[f]
            diff_dict[f"replica_{f}"] = nearest_replica[f]
            
        gt_vs_replica_diffs.append(diff_dict)
        
    diff_df = pd.DataFrame(gt_vs_replica_diffs)
    
    print(f"==================================================")
    print(f"   DIAGNOSTIC ANALYSIS: GT VS PERIODIC REPLICA   ")
    print(f"==================================================")
    print(f"Total Present Pairs: {total_present_count}")
    print(f"Pairs where GT was retrieved in Top-20: {success_gt_found_count} ({success_gt_found_count/total_present_count*100:.1f}%)")
    print(f"--------------------------------------------------")
    print(f"FEATURE COMPARISON (GT vs Nearest Periodic Replica):")
    print(f"{'Feature':<22} | {'GT Mean':<10} | {'Replica Mean':<12} | {'Delta (GT - Replica)':<20}")
    print(f"--------------------------------------------------")
    
    for f in features:
        gt_m = diff_df[f"gt_{f}"].mean()
        rep_m = diff_df[f"replica_{f}"].mean()
        delta_m = diff_df[f"delta_{f}"].mean()
        print(f"{f:<22} | {gt_m:<10.4f} | {rep_m:<12.4f} | {delta_m:<+20.4f}")
        
    print(f"==================================================")
    
    # Save detailed diagnostic CSV
    diff_df.to_csv("results/phase2/gt_vs_replica_analysis.csv", index=False)
    print("Detailed analysis saved to results/phase2/gt_vs_replica_analysis.csv")

if __name__ == "__main__":
    analyze_gt_vs_replica()
