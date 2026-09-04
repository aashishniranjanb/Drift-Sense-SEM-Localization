import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr

# Set random seed for determinism
np.random.seed(42)

def run_shadow_audit():
    print("==================================================================")
    print("       REJECTION_V2 SHADOW-ONLY AUDIT & THEORETICAL CEILING       ")
    print("==================================================================")

    # 1. Load Ground Truth and 180-Pair Features
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    pred_df = pd.read_csv("FINAL_SUBMISSION/validation/scale_only.csv")
    audit_df = pd.read_csv("data/phase2_dev/score_audit_180.csv")
    cache_df = pd.read_csv("FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv")

    # Merge into a comprehensive 180-pair evaluation dataframe
    dev_df = pd.merge(gt_df, pred_df, on="pair_id", suffixes=("_gt", "_pred"))
    dev_df = pd.merge(dev_df, cache_df, on="pair_id", suffixes=("", "_cache"))
    dev_df = pd.merge(dev_df, audit_df[["pair_id", "localization_error", "localization_tier", "correctness"]], on="pair_id")

    print(f"Loaded {len(dev_df)} development pairs.")

    # 2. Baseline Confusion Matrix (Competition convention: Absent is positive class for Rejection)
    # TP: gt_found == 0 & found == 0 (Absent correctly rejected)
    # FN: gt_found == 0 & found == 1 (Absent incorrectly accepted)
    # FP: gt_found == 1 & found == 0 (Present incorrectly rejected)
    # TN: gt_found == 1 & found == 1 (Present correctly accepted)
    tp_base = int(np.sum((dev_df["gt_found"] == 0) & (dev_df["found"] == 0)))
    fn_base = int(np.sum((dev_df["gt_found"] == 0) & (dev_df["found"] == 1)))
    fp_base = int(np.sum((dev_df["gt_found"] == 1) & (dev_df["found"] == 0)))
    tn_base = int(np.sum((dev_df["gt_found"] == 1) & (dev_df["found"] == 1)))

    prec_base = tp_base / (tp_base + fp_base)
    rec_base = tp_base / (tp_base + fn_base)
    f1_base = 2 * prec_base * rec_base / (prec_base + rec_base)
    rej_pts_base = f1_base * 15.0

    print(f"\n[Baseline Rejection Confusion]")
    print(f"TP (Absent Rejected)  : {tp_base} / 40")
    print(f"FN (Absent Accepted)  : {fn_base} / 40  (pair_140, pair_159)")
    print(f"FP (Present Rejected) : {fp_base} / 140")
    print(f"TN (Present Accepted) : {tn_base} / 140")
    print(f"Precision: {prec_base:.4f}, Recall: {rec_base:.4f}, F1: {f1_base:.4f}, Points: {rej_pts_base:.3f} / 15.0")

    # 3. Candidate Correctness on 180 pairs
    # Notice: Correct candidate means: gt_found == 1 AND raw localization_error <= 5.0
    dev_df["is_correct_candidate"] = ((dev_df["gt_found"] == 1) & (dev_df["localization_error"] <= 5.0)).astype(int)

    # Breakdown of candidate correctness across baseline decisions
    acc_correct = int(np.sum((dev_df["found"] == 1) & (dev_df["is_correct_candidate"] == 1)))
    acc_wrong = int(np.sum((dev_df["found"] == 1) & (dev_df["is_correct_candidate"] == 0)))
    rej_correct = int(np.sum((dev_df["found"] == 0) & (dev_df["is_correct_candidate"] == 1)))
    rej_wrong = int(np.sum((dev_df["found"] == 0) & (dev_df["is_correct_candidate"] == 0)))

    print(f"\n[Candidate Correctness Breakdown across Baseline Decisions]")
    print(f"Accepted (found=1, n={acc_correct + acc_wrong}): {acc_correct} correct (<=5px), {acc_wrong} wrong (absent: pair_140, pair_159)")
    print(f"Rejected (found=0, n={rej_correct + rej_wrong}): {rej_correct} correct (pair_027, pair_078), {rej_wrong} wrong (40 absent + 62 periodic replicas >5px)")

    # 4. Compute Theoretical Rejection Ceiling
    # Under the frozen candidate engine:
    # Max possible TP = 40 (all absent pairs rejected)
    # Min possible FN = 0
    # Min possible FP = 62 (because 62 present pairs have >5px error, accepting them destroys 40/40 loc)
    # Max possible TN = 78 (76 baseline + 2 recoverable pairs pair_027, pair_078)
    tp_ceil = 40
    fn_ceil = 0
    fp_ceil = 62
    tn_ceil = 78
    prec_ceil = tp_ceil / (tp_ceil + fp_ceil)
    rec_ceil = 1.0
    f1_ceil = 2 * prec_ceil * rec_ceil / (prec_ceil + rec_ceil)
    rej_pts_ceil = f1_ceil * 15.0

    print(f"\n[Theoretical Rejection Ceiling under Frozen Candidates (Preserving 40/40 Loc)]")
    print(f"Max TP: {tp_ceil}, Min FN: {fn_ceil}, Min FP: {fp_ceil}, Max TN: {tn_ceil}")
    print(f"Ceiling Precision: {prec_ceil:.4f}, Recall: {rec_ceil:.4f}, F1: {f1_ceil:.4f}")
    print(f"Ceiling Rejection Points: {rej_pts_ceil:.3f} / 15.000")
    print(f"Max Possible Rejection Delta: +{rej_pts_ceil - rej_pts_base:.3f} points")

    # 5. Synthetic Training Corpus Generation
    # Per user prompt: Generate synthetic dataset:
    # 5,000 genuine present
    # 5,000 periodic replicas (hard negatives with replica NCC ~ GT NCC)
    # 5,000 absent architecture negatives
    # 5,000 near-miss localization cases
    # Total = 20,000 synthetic samples
    print(f"\n[Generating 20,000 Synthetic Training & Validation Samples]...")
    rng = np.random.RandomState(42)

    features = ["top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]
    n_per_class = 5000

    # Class A: Genuine Present (High correlation, high margin, high neighborhood/context consistency)
    top1_score_A = rng.beta(8, 2, n_per_class) * 0.3 + 0.7       # 0.70 - 1.00
    margin_A = rng.exponential(0.06, n_per_class) + 0.03          # 0.03 - 0.25
    top1_corr_A = rng.beta(7, 2, n_per_class) * 0.3 + 0.7        # 0.70 - 0.98
    top1_ctx_A = rng.beta(6, 2, n_per_class) * 0.4 + 0.55        # 0.55 - 0.95
    top1_neigh_A = rng.beta(6, 2, n_per_class) * 0.4 + 0.55      # 0.55 - 0.95
    top1_grad_A = rng.beta(7, 2, n_per_class) * 0.3 + 0.65       # 0.65 - 0.95
    mode_strong_A = rng.choice([1, 2, 3], size=n_per_class, p=[0.7, 0.2, 0.1])
    y_A = np.ones(n_per_class, dtype=int)

    # Class B: Periodic Replicas (High correlation, near-zero margin, low neighborhood consistency)
    top1_score_B = rng.beta(7, 3, n_per_class) * 0.3 + 0.65      # 0.65 - 0.92
    margin_B = rng.exponential(0.008, n_per_class)               # 0.001 - 0.02 (very low gap)
    top1_corr_B = rng.beta(7, 3, n_per_class) * 0.3 + 0.65       # High raw peak
    top1_ctx_B = rng.beta(3, 4, n_per_class) * 0.5 + 0.2         # Mediocre context
    top1_neigh_B = rng.beta(2, 5, n_per_class) * 0.4 + 0.1       # Low neighborhood agreement
    top1_grad_B = rng.beta(4, 4, n_per_class) * 0.4 + 0.3        # Weak gradient agreement
    mode_strong_B = rng.choice([3, 4, 5, 6], size=n_per_class)   # Multiple periodic peaks
    y_B = np.zeros(n_per_class, dtype=int)

    # Class C: Absent Architecture Negatives (Low/moderate correlation, low context/neigh)
    top1_score_C = rng.beta(3, 5, n_per_class) * 0.4 + 0.3       # 0.30 - 0.65
    margin_B_C = rng.exponential(0.02, n_per_class)
    top1_corr_C = rng.beta(3, 5, n_per_class) * 0.4 + 0.3
    top1_ctx_C = rng.beta(2, 5, n_per_class) * 0.4 + 0.1
    top1_neigh_C = rng.beta(2, 5, n_per_class) * 0.4 + 0.1
    top1_grad_C = rng.beta(2, 5, n_per_class) * 0.4 + 0.1
    mode_strong_C = rng.choice([1, 2, 3], size=n_per_class)
    y_C = np.zeros(n_per_class, dtype=int)

    # Class D: Near-Miss Localization Cases (>5px error on same structure)
    top1_score_D = rng.beta(5, 4, n_per_class) * 0.3 + 0.55      # 0.55 - 0.82
    margin_D = rng.exponential(0.015, n_per_class)
    top1_corr_D = rng.beta(5, 4, n_per_class) * 0.3 + 0.55
    top1_ctx_D = rng.beta(4, 4, n_per_class) * 0.4 + 0.3
    top1_neigh_D = rng.beta(3, 4, n_per_class) * 0.4 + 0.2
    top1_grad_D = rng.beta(4, 4, n_per_class) * 0.4 + 0.3
    mode_strong_D = rng.choice([2, 3, 4], size=n_per_class)
    y_D = np.zeros(n_per_class, dtype=int)

    X_synth = np.vstack([
        np.column_stack([top1_score_A, margin_A, top1_corr_A, top1_ctx_A, top1_neigh_A, top1_grad_A, mode_strong_A]),
        np.column_stack([top1_score_B, margin_B, top1_corr_B, top1_ctx_B, top1_neigh_B, top1_grad_B, mode_strong_B]),
        np.column_stack([top1_score_C, margin_B_C, top1_corr_C, top1_ctx_C, top1_neigh_C, top1_grad_C, mode_strong_C]),
        np.column_stack([top1_score_D, margin_D, top1_corr_D, top1_ctx_D, top1_neigh_D, top1_grad_D, mode_strong_D]),
    ])
    y_synth = np.concatenate([y_A, y_B, y_C, y_D])

    # Shuffle synthetic data
    perm = rng.permutation(len(y_synth))
    X_synth, y_synth = X_synth[perm], y_synth[perm]

    # Split: 70% train (14,000), 30% held-out test (6,000)
    split_idx = int(0.70 * len(y_synth))
    X_train, y_train = X_synth[:split_idx], y_synth[:split_idx]
    X_test, y_test = X_synth[split_idx:], y_synth[split_idx:]

    print(f"Synthetic dataset: {len(X_train)} train, {len(X_test)} held-out validation.")

    # 6. Train Models
    print(f"\n[Training Model A: Logistic Regression]...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    p_test_lr = lr.predict_proba(X_test)[:, 1]

    auc_test_lr = roc_auc_score(y_test, p_test_lr)
    ap_test_lr = average_precision_score(y_test, p_test_lr)
    brier_test_lr = brier_score_loss(y_test, p_test_lr)
    print(f"Logistic Regression Held-Out: ROC AUC = {auc_test_lr:.4f}, PR AUC = {ap_test_lr:.4f}, Brier = {brier_test_lr:.4f}")

    print(f"\n[Training Model B: HistGradientBoosting (depth=2)]...")
    hgb = HistGradientBoostingClassifier(max_depth=2, random_state=42)
    hgb.fit(X_train, y_train)
    p_test_hgb = hgb.predict_proba(X_test)[:, 1]

    auc_test_hgb = roc_auc_score(y_test, p_test_hgb)
    ap_test_hgb = average_precision_score(y_test, p_test_hgb)
    brier_test_hgb = brier_score_loss(y_test, p_test_hgb)
    print(f"HistGradientBoosting Held-Out: ROC AUC = {auc_test_hgb:.4f}, PR AUC = {ap_test_hgb:.4f}, Brier = {brier_test_hgb:.4f}")

    # 7. Evaluate on 180-Pair Dev Set (SHADOW DIAGNOSTIC ONLY)
    X_dev = dev_df[features].values.astype(float)
    y_dev = dev_df["is_correct_candidate"].values.astype(int)

    p_dev_lr = lr.predict_proba(X_dev)[:, 1]
    p_dev_hgb = hgb.predict_proba(X_dev)[:, 1]

    auc_dev_lr = roc_auc_score(y_dev, p_dev_lr)
    auc_dev_hgb = roc_auc_score(y_dev, p_dev_hgb)
    print(f"\n[180-Pair Dev Set Shadow Candidate Correctness AUC]")
    print(f"LR  P(correct) AUC: {auc_dev_lr:.4f}")
    print(f"HGB P(correct) AUC: {auc_dev_hgb:.4f}")

    dev_df["p_correct_lr"] = p_dev_lr
    dev_df["p_correct_hgb"] = p_dev_hgb

    # Score distributions across populations
    print(f"\n[HGB P(correct) Score Distribution across True Populations]:")
    for group, desc in [
        ((dev_df["found"] == 1) & (dev_df["is_correct_candidate"] == 1), "Correct Accepted (True Positives)"),
        ((dev_df["found"] == 1) & (dev_df["is_correct_candidate"] == 0), "Incorrect Accepted (False Accepts - Absent)"),
        ((dev_df["found"] == 0) & (dev_df["is_correct_candidate"] == 1), "Correct Rejected (Recoverable FN - 2 pairs)"),
        ((dev_df["found"] == 0) & (dev_df["is_correct_candidate"] == 0), "Incorrect Rejected (Absent + Periodic Replicas)")
    ]:
        subset = dev_df[group]["p_correct_hgb"]
        print(f"  {desc:<45s}: n={len(subset):2d}, mean={subset.mean():.3f}, min={subset.min():.3f}, max={subset.max():.3f}")

    # 8. SHADOW INTERVENTION A: ACCEPTED-CANDIDATE VETO
    # For currently found=1 pairs (n=78):
    # Veto if p_correct < T_veto -> changes found from 1 to 0
    print(f"\n[SHADOW INTERVENTION A: ACCEPTED-CANDIDATE VETO SWEEP (HGB)]")
    veto_thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    veto_results = []

    for t in veto_thresholds:
        # Veto condition
        veto_mask = (dev_df["found"] == 1) & (dev_df["p_correct_hgb"] < t)
        n_veto = int(np.sum(veto_mask))

        # Check what got vetoed
        vetoed_correct = int(np.sum(veto_mask & (dev_df["is_correct_candidate"] == 1)))
        vetoed_wrong = int(np.sum(veto_mask & (dev_df["is_correct_candidate"] == 0)))

        # New found vector
        sim_found = dev_df["found"].copy()
        sim_found[veto_mask] = 0

        # Calculate new confusion
        tp_sim = int(np.sum((dev_df["gt_found"] == 0) & (sim_found == 0)))
        fn_sim = int(np.sum((dev_df["gt_found"] == 0) & (sim_found == 1)))
        fp_sim = int(np.sum((dev_df["gt_found"] == 1) & (sim_found == 0)))
        tn_sim = int(np.sum((dev_df["gt_found"] == 1) & (sim_found == 1)))

        prec_sim = tp_sim / (tp_sim + fp_sim) if (tp_sim + fp_sim) > 0 else 0
        rec_sim = tp_sim / (tp_sim + fn_sim) if (tp_sim + fn_sim) > 0 else 0
        f1_sim = 2 * prec_sim * rec_sim / (prec_sim + rec_sim) if (prec_sim + rec_sim) > 0 else 0
        rej_pts_sim = f1_sim * 15.0

        # Localization check:
        # If any correct candidate is vetoed, does localization drop?
        # Set A and Set B localized accuracy
        set_a = dev_df[(dev_df["set_type"] == "SetA") & (dev_df["gt_found"] == 1)]
        set_b = dev_df[(dev_df["set_type"] == "SetB") & (dev_df["gt_found"] == 1)]

        loc_a = np.mean(set_a["localization_error"][sim_found.loc[set_a.index] == 1] <= 5.0) * 100.0 if np.sum(sim_found.loc[set_a.index] == 1) > 0 else 0.0
        loc_b = np.mean(set_b["localization_error"][sim_found.loc[set_b.index] == 1] <= 5.0) * 100.0 if np.sum(sim_found.loc[set_b.index] == 1) > 0 else 0.0
        loc_pts_sim = (0.45 * loc_a + 0.55 * loc_b) * 0.40

        d_rej = rej_pts_sim - rej_pts_base
        d_loc = loc_pts_sim - 40.0
        d_tot = d_rej + d_loc

        veto_results.append({
            "Threshold": t,
            "Total Vetoed": n_veto,
            "Correct Removed": vetoed_correct,
            "Wrong Removed": vetoed_wrong,
            "Loc Pts": loc_pts_sim,
            "Rej Pts": rej_pts_sim,
            "Rej Delta": d_rej,
            "Total Delta": d_tot
        })
        print(f"  T_veto = {t:0.2f}: Vetoed={n_veto:2d} (Correct={vetoed_correct}, Wrong={vetoed_wrong}) | Loc={loc_pts_sim:.2f}, Rej={rej_pts_sim:.3f} (delta={d_rej:+.3f}), Net delta={d_tot:+.3f}")

    # 9. SHADOW INTERVENTION B: REJECTED-CANDIDATE RESCUE
    # For currently found=0 pairs (n=102):
    # Rescue if p_correct > T_rescue -> changes found from 0 to 1
    print(f"\n[SHADOW INTERVENTION B: REJECTED-CANDIDATE RESCUE SWEEP (HGB)]")
    rescue_thresholds = [0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99]
    rescue_results = []

    for t in rescue_thresholds:
        # Rescue condition
        rescue_mask = (dev_df["found"] == 0) & (dev_df["p_correct_hgb"] > t)
        n_rescued = int(np.sum(rescue_mask))

        rescued_correct = int(np.sum(rescue_mask & (dev_df["is_correct_candidate"] == 1)))
        rescued_wrong = int(np.sum(rescue_mask & (dev_df["is_correct_candidate"] == 0)))

        sim_found = dev_df["found"].copy()
        sim_found[rescue_mask] = 1

        # Rejection metrics
        tp_sim = int(np.sum((dev_df["gt_found"] == 0) & (sim_found == 0)))
        fn_sim = int(np.sum((dev_df["gt_found"] == 0) & (sim_found == 1)))
        fp_sim = int(np.sum((dev_df["gt_found"] == 1) & (sim_found == 0)))
        tn_sim = int(np.sum((dev_df["gt_found"] == 1) & (sim_found == 1)))

        prec_sim = tp_sim / (tp_sim + fp_sim) if (tp_sim + fp_sim) > 0 else 0
        rec_sim = tp_sim / (tp_sim + fn_sim) if (tp_sim + fn_sim) > 0 else 0
        f1_sim = 2 * prec_sim * rec_sim / (prec_sim + rec_sim) if (prec_sim + rec_sim) > 0 else 0
        rej_pts_sim = f1_sim * 15.0

        # Localization metrics:
        set_a = dev_df[(dev_df["set_type"] == "SetA") & (dev_df["gt_found"] == 1)]
        set_b = dev_df[(dev_df["set_type"] == "SetB") & (dev_df["gt_found"] == 1)]

        loc_a = np.mean(set_a["localization_error"][sim_found.loc[set_a.index] == 1] <= 5.0) * 100.0 if np.sum(sim_found.loc[set_a.index] == 1) > 0 else 0.0
        loc_b = np.mean(set_b["localization_error"][sim_found.loc[set_b.index] == 1] <= 5.0) * 100.0 if np.sum(sim_found.loc[set_b.index] == 1) > 0 else 0.0
        loc_pts_sim = (0.45 * loc_a + 0.55 * loc_b) * 0.40

        d_rej = rej_pts_sim - rej_pts_base
        d_loc = loc_pts_sim - 40.0
        d_tot = d_rej + d_loc

        rescue_results.append({
            "Threshold": t,
            "Total Rescued": n_rescued,
            "Correct Rescued": rescued_correct,
            "Wrong Rescued": rescued_wrong,
            "Loc Pts": loc_pts_sim,
            "Rej Pts": rej_pts_sim,
            "Rej Delta": d_rej,
            "Loc Delta": d_loc,
            "Total Delta": d_tot
        })
        print(f"  T_rescue = {t:0.2f}: Rescued={n_rescued:2d} (Correct={rescued_correct}, Wrong={rescued_wrong}) | Loc={loc_pts_sim:.2f} (delta={d_loc:+.2f}), Rej={rej_pts_sim:.3f} (delta={d_rej:+.3f}), Net delta={d_tot:+.3f}")

    return {
        "base_confusion": (tp_base, fn_base, fp_base, tn_base, f1_base, rej_pts_base),
        "ceiling": (tp_ceil, fn_ceil, fp_ceil, tn_ceil, f1_ceil, rej_pts_ceil),
        "synth_eval": (auc_test_lr, ap_test_lr, auc_test_hgb, ap_test_hgb),
        "dev_auc": (auc_dev_lr, auc_dev_hgb),
        "veto_results": veto_results,
        "rescue_results": rescue_results
    }

if __name__ == "__main__":
    run_shadow_audit()
