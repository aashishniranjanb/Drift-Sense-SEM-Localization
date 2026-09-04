import os
import sys
import numpy as np
import pandas as pd

def main():
    print("==================================================================")
    print("      RERANK-V1 SHADOW EVALUATION (TOP-200 CANDIDATES)            ")
    print("==================================================================")

    # We want to test whether multi-signal structural re-ranking can rescue
    # the 26 ranking failure pairs WITHOUT breaking any of the 76 successes.

    forensics_26 = pd.read_csv("FINAL_SUBMISSION/validation/ranking_failures_26_forensics.csv")
    print(f"Loaded {len(forensics_26)} ranking failure forensics.")
    print("Average features for GT vs Replica in the 26 failure pairs:")
    print(f"  GT corr: {forensics_26['corr_gt'].mean():.4f} vs Replica: {forensics_26['corr_top1'].mean():.4f}")
    print(f"  GT ctx : {forensics_26['ctx_gt'].mean():.4f} vs Replica: {forensics_26['ctx_top1'].mean():.4f}")
    print(f"  GT neigh: {forensics_26['neigh_gt'].mean():.4f} vs Replica: {forensics_26['neigh_top1'].mean():.4f}")
    print(f"  GT grad: {forensics_26['grad_gt'].mean():.4f} vs Replica: {forensics_26['grad_top1'].mean():.4f}")

    # Notice: In the 26 failures:
    # Replica leads in corr by 0.032 on average!
    # But GT leads in context or neighborhood in 46.2% of pairs!
    # What if we apply a Second-Look Evaluator for close candidates?
    # User's exact instruction:
    # "8. Add a 'second-look' evaluator only for close competitors
    # Don't spend computation on all 200 candidates equally.
    # Top candidate -> if margin > safe threshold -> keep
    # Otherwise: top1 ~ top2 -> expensive forensic comparison
    # This is a classic coarse-to-fine strategy."

if __name__ == "__main__":
    main()
