import os
import sys
import numpy as np
import pandas as pd

def main():
    # Load candidate pool audit
    pool_audit = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    score_audit = pd.read_csv("data/phase2_dev/score_audit_180.csv")
    cache = pd.read_csv("FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv")

    df = pd.merge(pool_audit, score_audit[["pair_id", "top1_score", "top2_score", "top5_score"]], on="pair_id")
    df = pd.merge(df, cache[["pair_id", "margin", "top1_corr"]], on="pair_id")

    print("==================================================================")
    print("      GT VS BEST-REPLICA MARGIN & DIFFICULTY DECOMPOSITION        ")
    print("==================================================================")

    # For SUCCESS_ACCEPTED (n=76): Top-1 is GT candidate!
    # Margin = Top-1 - Top-2
    successes = df[df["category"] == "SUCCESS_ACCEPTED"].copy()
    successes["gt_margin"] = successes["top1_score"] - successes["top2_score"]

    print(f"\n1. SUCCESSFUL ACCEPTANCES (n={len(successes)}):")
    print(f"   GT wins by > 0.10      : {sum(successes['gt_margin'] > 0.10)} pairs")
    print(f"   GT wins by 0.05 - 0.10 : {sum((successes['gt_margin'] <= 0.10) & (successes['gt_margin'] > 0.05))} pairs")
    print(f"   GT wins by 0.01 - 0.05 : {sum((successes['gt_margin'] <= 0.05) & (successes['gt_margin'] > 0.01))} pairs")
    print(f"   GT wins by < 0.01      : {sum(successes['gt_margin'] <= 0.01)} pairs")
    print(f"   Median GT victory margin: {successes['gt_margin'].median():.4f}")

    # For RANKING_FAILURES (n=26):
    # GT candidate is in the pool, but lost to Top-1 (periodic replica)!
    ranking_fails = df[df["category"] == "RANKING_FAILURE"].copy()
    print(f"\n2. RANKING FAILURES (n={len(ranking_fails)}):")
    print(f"   GT candidate was in pool (error <= 5px), but lost rank 1 to a periodic replica.")
    print(f"   Replica victory margin (Top1 - Top2): median={ranking_fails['margin'].median():.4f}")

    # For RETRIEVAL_FAILURES (n=35):
    retrieval_fails = df[df["category"] == "RETRIEVAL_FAILURE"].copy()
    print(f"\n3. RETRIEVAL FAILURES (n={len(retrieval_fails)}):")
    print(f"   GT candidate did NOT enter the 200-candidate pool.")
    print(f"   Near-retrieval (5 - 10 px) : {sum((retrieval_fails['min_pool_err'] > 5) & (retrieval_fails['min_pool_err'] <= 10))} pairs")
    print(f"   Mid-retrieval (10 - 25 px) : {sum((retrieval_fails['min_pool_err'] > 10) & (retrieval_fails['min_pool_err'] <= 25))} pairs")
    print(f"   Far-retrieval (> 25 px)    : {sum(retrieval_fails['min_pool_err'] > 25)} pairs")

    # For REJECTION_FAILURES (n=3):
    rej_fails = df[df["category"] == "REJECTION_FAILURE"].copy()
    print(f"\n4. REJECTION FAILURES (n={len(rej_fails)}):")
    print(f"   GT candidate won rank 1, but was rejected by V28-C gate:")
    for _, r in rej_fails.iterrows():
        print(f"     {r['pair_id']} ({r['set_type']}): top1_score={r['top1_score']:.4f}, margin={r['margin']:.4f}, error={r['top1_err']:.2f}px")

if __name__ == "__main__":
    main()
