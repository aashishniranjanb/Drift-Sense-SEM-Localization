import os
import sys
import numpy as np
import pandas as pd

def main():
    print("==================================================================")
    print("      RERANK-V1 SHADOW RE-RANKING EXPERIMENT (TOP-200 POOL)       ")
    print("==================================================================")

    # 1. Load the 26 ranking failure forensics
    df_26 = pd.read_csv("FINAL_SUBMISSION/validation/ranking_failures_26_forensics.csv")
    print(f"Loaded 26 ranking failures.")

    # In df_26, we have:
    # corr_gt, corr_top1, d_corr (gt - top1)
    # ctx_gt, ctx_top1, d_ctx (gt - top1)
    # neigh_gt, neigh_top1, d_neigh (gt - top1)
    # grad_gt, grad_top1, d_grad (gt - top1)

    # Let's test linear re-ranking weights:
    # S(c) = corr + w_ctx * ctx + w_neigh * neigh + w_grad * grad
    # We want: S(gt) > S(top1) for as many of the 26 as possible
    # Delta S = d_corr + w_ctx * d_ctx + w_neigh * d_neigh + w_grad * d_grad > 0

    print("\n[Grid Search of Structural Weights on the 26 Failure Forensics]:")
    best_rescued = 0
    best_weights = None

    for w_ctx in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        for w_neigh in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for w_grad in [0.0, 0.1, 0.2]:
                dS = df_26["d_corr (gt - top1)"] + w_ctx * df_26["d_ctx (gt - top1)"] + w_neigh * df_26["d_neigh (gt - top1)"] + w_grad * df_26["d_grad (gt - top1)"]
                n_rescued = sum(dS > 0)
                if n_rescued > best_rescued:
                    best_rescued = n_rescued
                    best_weights = (w_ctx, w_neigh, w_grad)

    print(f"Best linear combination rescues: {best_rescued} / 26 failures!")
    print(f"Optimal weights: w_ctx={best_weights[0]}, w_neigh={best_weights[1]}, w_grad={best_weights[2]}")

    # Show which pairs are rescued
    w_ctx, w_neigh, w_grad = best_weights
    df_26["dS"] = df_26["d_corr (gt - top1)"] + w_ctx * df_26["d_ctx (gt - top1)"] + w_neigh * df_26["d_neigh (gt - top1)"] + w_grad * df_26["d_grad (gt - top1)"]
    rescued_pairs = df_26[df_26["dS"] > 0]["pair_id"].tolist()
    print(f"\nRescued pairs ({len(rescued_pairs)}): {rescued_pairs}")

if __name__ == "__main__":
    main()
