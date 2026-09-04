import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
from periodicity_detector import estimate_periodicity_from_corr
import rerank_features

np.random.seed(42)

def generate_adversarial_synthetic_pairs(n_pairs=10000, seed=42):
    """
    Generates synthetic adversarial pairwise comparisons (Candidate A vs Candidate B)
    where A is the true match and B is a periodic replica.
    Crucially includes cases where corr(replica) >= corr(true).
    """
    rng = np.random.RandomState(seed)
    feature_names = [
        "d_corr", "d_psr", "d_grad", "d_phase", "d_ctx", "d_neigh",
        "d_prominence", "d_sharpness", "d_lattice_res", "d_nearest_dist"
    ]

    X = []
    y = []

    for i in range(n_pairs // 2):
        # 50% of the time, replica has HIGHER raw correlation (adversarial Level 4)
        is_adversarial = rng.rand() < 0.55

        if is_adversarial:
            # Replica has higher correlation by 0.005 to 0.045
            d_corr = -rng.uniform(0.005, 0.045)
            # But true candidate has higher macro-context
            d_ctx = rng.uniform(0.010, 0.090)
            # True candidate has higher neighborhood consistency
            d_neigh = rng.uniform(0.005, 0.070)
            # Gradient difference is small or favors GT
            d_grad = rng.uniform(-0.030, 0.050)
            # Phase penalty favors GT (lower penalty for GT means negative delta_penalty)
            d_phase = -rng.uniform(0.0, 0.08)
            # True candidate has sharper peak morphology
            d_prominence = rng.uniform(0.005, 0.040)
            d_sharpness = rng.uniform(0.010, 0.080)
            d_lat_res = -rng.uniform(0.05, 0.20) # GT has lower lattice residual
            d_near_dist = rng.uniform(-10.0, 30.0)
            d_psr = rng.uniform(-0.5, 1.5)
        else:
            # Standard case: GT is slightly ahead across all signals
            d_corr = rng.uniform(0.002, 0.030)
            d_ctx = rng.uniform(0.005, 0.050)
            d_neigh = rng.uniform(0.002, 0.040)
            d_grad = rng.uniform(0.001, 0.040)
            d_phase = -rng.uniform(0.0, 0.05)
            d_prominence = rng.uniform(0.002, 0.030)
            d_sharpness = rng.uniform(0.005, 0.050)
            d_lat_res = -rng.uniform(0.02, 0.15)
            d_near_dist = rng.uniform(-5.0, 20.0)
            d_psr = rng.uniform(0.2, 2.0)

        vec_A_over_B = np.array([
            d_corr, d_psr, d_grad, d_phase, d_ctx, d_neigh,
            d_prominence, d_sharpness, d_lat_res, d_near_dist
        ])

        # Pair 1: A vs B (GT vs Replica, label = 1)
        X.append(vec_A_over_B)
        y.append(1)

        # Pair 2: B vs A (Replica vs GT, label = 0, exact antisymmetric vector)
        X.append(-vec_A_over_B)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    perm = rng.permutation(len(y))
    return X[perm], y[perm], feature_names

def main():
    print("==================================================================", flush=True)
    print("   RERANK-V2 COMPREHENSIVE SHADOW EXPERIMENT & BENCHMARK AUDIT    ", flush=True)
    print("==================================================================", flush=True)

    # 1. Generate Deterministic Synthetic Adversarial Training Set (Step 4)
    print("\n[STEP 4: Generating 10,000 Synthetic Adversarial Pairwise Samples]...", flush=True)
    X_synth, y_synth, feat_names = generate_adversarial_synthetic_pairs(n_pairs=10000, seed=42)

    # 70% train, 30% held-out test
    split = int(0.70 * len(y_synth))
    X_train, y_train = X_synth[:split], y_synth[:split]
    X_test, y_test = X_synth[split:], y_synth[split:]

    print(f"Synthetic Pairwise Dataset: {len(X_train)} train, {len(X_test)} validation.", flush=True)

    # 2. Train Models (Step 5)
    print("\n[STEP 5: Training & Comparing Models on Adversarial Distribution]...", flush=True)

    # Model A: Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, p_lr)
    brier_lr = brier_score_loss(y_test, p_lr)
    print(f"  Model A: LogisticRegression    -> Held-Out AUC = {auc_lr:.4f}, Brier = {brier_lr:.4f}", flush=True)

    # Model B: HistGradientBoosting (depth=2)
    hgb2 = HistGradientBoostingClassifier(max_depth=2, random_state=42)
    hgb2.fit(X_train, y_train)
    p_hgb2 = hgb2.predict_proba(X_test)[:, 1]
    auc_hgb2 = roc_auc_score(y_test, p_hgb2)
    brier_hgb2 = brier_score_loss(y_test, p_hgb2)
    print(f"  Model B: HistGradientBoosting (depth=2) -> Held-Out AUC = {auc_hgb2:.4f}, Brier = {brier_hgb2:.4f}", flush=True)

    # Model C: HistGradientBoosting (depth=3)
    hgb3 = HistGradientBoostingClassifier(max_depth=3, random_state=42)
    hgb3.fit(X_train, y_train)
    p_hgb3 = hgb3.predict_proba(X_test)[:, 1]
    auc_hgb3 = roc_auc_score(y_test, p_hgb3)
    brier_hgb3 = brier_score_loss(y_test, p_hgb3)
    print(f"  Model C: HistGradientBoosting (depth=3) -> Held-Out AUC = {auc_hgb3:.4f}, Brier = {brier_hgb3:.4f}", flush=True)

    print("\nLogistic Regression Coefficients:", flush=True)
    for fn, coef in zip(feat_names, lr.coef_[0]):
        print(f"  {fn:<20s}: {coef:+.4f}", flush=True)

    # Save models for shadow evaluation
    os.makedirs("FINAL_SUBMISSION/runtime/models", exist_ok=True)
    with open("FINAL_SUBMISSION/runtime/models/rerank_v2_lr.pkl", "wb") as f:
        pickle.dump({"model": lr, "features": feat_names}, f)
    with open("FINAL_SUBMISSION/runtime/models/rerank_v2_hgb2.pkl", "wb") as f:
        pickle.dump({"model": hgb2, "features": feat_names}, f)

    # 3. Forensic Test on the 26 Ranking Failures (Step 6)
    print("\n" + "="*65, flush=True)
    print(" [STEP 6: Forensic Test on the 26 Ranking Failure Pairs]", flush=True)
    print("="*65, flush=True)

    forensics_26 = pd.read_csv("FINAL_SUBMISSION/validation/ranking_failures_26_forensics.csv")
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    raw_v25 = pd.read_csv("data/phase2_dev/v25_predictions.csv")

    with open("FINAL_SUBMISSION/runtime/models/ranker.pkl", "rb") as f:
        v25_ranker = pickle.load(f)

    def extract_full_candidates(pid):
        row = gt_df[gt_df["pair_id"] == pid].iloc[0]
        v25_row = raw_v25[raw_v25["pair_id"] == pid].iloc[0]
        gt_x, gt_y = row["gt_x"], row["gt_y"]
        est_scale = float(v25_row["scale"])
        est_theta = float(v25_row["theta"])

        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)

        tw, th = int(round(ref.shape[1] / est_scale)), int(round(ref.shape[0] / est_scale))
        tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        if abs(est_theta) > 0.01:
            tpl_rot = rotate_image(tpl, est_theta)
        else:
            tpl_rot = tpl
        corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)
        cands = extract_candidates_akhilesh(corr_plane, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)
        cands = cluster_replica_families(cands, est_scale)

        per = estimate_periodicity_from_corr(corr_plane)
        pitch_x, pitch_y = per["pitch_x"], per["pitch_y"]

        rows = []
        for idx, c in enumerate(cands):
            cx, cy = c["cx"], c["cy"]
            px, py = c["peak_x"], c["peak_y"]
            ctx = verify_candidate_context(ref, srch, cx, cy, est_scale, est_theta)
            phase_pen = verify_phase_consistency(srch, tpl_rot, px, py)
            neigh = compute_neighborhood_consistency(srch, tpl_rot, px, py, pitch_x, pitch_y)
            gncc = compute_gradient_ncc(srch, tpl_rot, px, py)
            morph = rerank_features.compute_candidate_morphology(corr_plane, px, py)
            comp = rerank_features.compute_competitor_features(cands, idx, pitch_x, pitch_y)

            rows.append({
                "cx": cx, "cy": cy, "peak_x": px, "peak_y": py,
                "corr_score": c["corr_score"], "psr": c.get("psr", 0),
                "context_128": ctx["s128"], "context_combined": ctx["combined"],
                "phase_penalty": phase_pen, "family_population": c.get("family_population", 1),
                "dist_to_center": c.get("dist_to_center", 0.0),
                "neigh_cons": neigh, "grad_ncc": gncc,
                "prominence": morph["prominence"], "curvature": morph["curvature"], "sharpness": morph["sharpness"],
                "nearest_competitor_dist": comp["nearest_competitor_dist"],
                "lattice_residual": comp["lattice_residual"],
                "err_to_gt": float(np.hypot(cx - gt_x, cy - gt_y))
            })

        df = pd.DataFrame(rows)
        fcols = ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
                 "dist_to_center", "neigh_cons", "grad_ncc"]
        for col in fcols:
            df[col + "_rel"] = df[col] - df[col].median()
        df["family_ratio"] = df["family_population"] / len(cands)

        # Legacy ML score
        df["v25_ml_score"] = v25_ranker["model"].predict_proba(df[v25_ranker["features"]])[:, 1]
        df_sorted = df.sort_values(by="v25_ml_score", ascending=False).reset_index(drop=True)
        return df_sorted, pitch_x, pitch_y

    def evaluate_reranker_on_candidates(df_cands, model, mode="lr"):
        """
        Ranks candidates using pairwise comparison against the top-5 competitors.
        Returns the re-ranked dataframe.
        """
        # We only need to re-evaluate the top 10 candidates to keep runtime fast and safe
        top_k_pool = min(10, len(df_cands))
        pool = df_cands.iloc[:top_k_pool].copy()

        pairwise_scores = np.zeros(top_k_pool)

        for i in range(top_k_pool):
            ci = pool.iloc[i]
            for j in range(top_k_pool):
                if i == j:
                    continue
                cj = pool.iloc[j]

                d_vec = np.array([[
                    ci["corr_score"] - cj["corr_score"],
                    ci["psr"] - cj["psr"],
                    ci["grad_ncc"] - cj["grad_ncc"],
                    ci["phase_penalty"] - cj["phase_penalty"],
                    ci["context_combined"] - cj["context_combined"],
                    ci["neigh_cons"] - cj["neigh_cons"],
                    ci["prominence"] - cj["prominence"],
                    ci["sharpness"] - cj["sharpness"],
                    ci["lattice_residual"] - cj["lattice_residual"],
                    ci["nearest_competitor_dist"] - cj["nearest_competitor_dist"]
                ]])

                prob_i_wins = model.predict_proba(d_vec)[0, 1]
                pairwise_scores[i] += prob_i_wins

        pool["rerank_score"] = pairwise_scores
        # Sort by rerank score descending
        pool_reranked = pool.sort_values(by="rerank_score", ascending=False).reset_index(drop=True)
        return pool_reranked

    print(f"Evaluating LR and HGB2 pairwise re-rankers on the 26 ranking failures...", flush=True)

    rescued_lr = 0
    rescued_hgb = 0
    rescue_details = []

    for pid in forensics_26["pair_id"]:
        df_cands, px, py = extract_full_candidates(pid)

        # Baseline Top-1 error
        base_err = df_cands.iloc[0]["err_to_gt"]
        base_margin = df_cands.iloc[0]["v25_ml_score"] - df_cands.iloc[1]["v25_ml_score"] if len(df_cands) > 1 else 1.0

        # RERANK with LR
        rerank_lr = evaluate_reranker_on_candidates(df_cands, lr, mode="lr")
        new_err_lr = rerank_lr.iloc[0]["err_to_gt"]

        # RERANK with HGB2
        rerank_hgb = evaluate_reranker_on_candidates(df_cands, hgb2, mode="hgb")
        new_err_hgb = rerank_hgb.iloc[0]["err_to_gt"]

        if new_err_lr <= 5.0 and base_err > 5.0:
            rescued_lr += 1
        if new_err_hgb <= 5.0 and base_err > 5.0:
            rescued_hgb += 1

        rescue_details.append({
            "pair_id": pid,
            "base_err": base_err,
            "base_margin": base_margin,
            "new_err_lr": new_err_lr,
            "new_err_hgb": new_err_hgb,
            "rescued_lr": int(new_err_lr <= 5.0),
            "rescued_hgb": int(new_err_hgb <= 5.0)
        })

    print(f"\n26 Failure Set Results:", flush=True)
    print(f"  Logistic Regression Pairwise: {rescued_lr} / 26 failures rescued to <=5px!", flush=True)
    print(f"  HistGradientBoosting (depth=2): {rescued_hgb} / 26 failures rescued to <=5px!", flush=True)

    df_rescue_summary = pd.DataFrame(rescue_details)
    print(df_rescue_summary[["pair_id", "base_err", "base_margin", "new_err_lr", "rescued_lr"]].head(10), flush=True)

    # 4. Mandatory Safety Test on 76 Successful Acceptances (Step 7)
    print("\n" + "="*65, flush=True)
    print(" [STEP 7: Mandatory Safety Test on the 76 Successful Pairs]", flush=True)
    print("="*65, flush=True)

    pool_audit = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    success_pids = pool_audit[pool_audit["category"] == "SUCCESS_ACCEPTED"]["pair_id"].tolist()

    safety_records = []
    broken_count = 0

    for pid in success_pids:
        df_cands, px, py = extract_full_candidates(pid)
        base_err = df_cands.iloc[0]["err_to_gt"]
        base_margin = df_cands.iloc[0]["v25_ml_score"] - df_cands.iloc[1]["v25_ml_score"] if len(df_cands) > 1 else 1.0

        rerank_lr = evaluate_reranker_on_candidates(df_cands, lr, mode="lr")
        new_err = rerank_lr.iloc[0]["err_to_gt"]

        is_broken = int(base_err <= 5.0 and new_err > 5.0)
        if is_broken:
            broken_count += 1

        safety_records.append({
            "pair_id": pid,
            "base_err": base_err,
            "base_margin": base_margin,
            "new_err": new_err,
            "is_broken": is_broken
        })

    print(f"Raw Re-ranker Safety Result on 76 Successes: {broken_count} broken out of 76.", flush=True)

    # 5. Second-Look Margin Policy Sweep (Step 8)
    print("\n" + "="*65, flush=True)
    print(" [STEP 8: Second-Look Margin Policy Sweep]", flush=True)
    print("="*65, flush=True)

    # Combine the 26 failures + 76 successes (102 pairs total)
    df_safety = pd.DataFrame(safety_records)

    sweep_thresholds = [0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100]
    print(f"{'Threshold':<10s} | {'Rescued (out of 26)':<20s} | {'Broken (out of 76)':<20s} | {'Net Loc Gain':<15s}", flush=True)
    print("-" * 75, flush=True)

    best_thresh = 0.0
    max_net = -999

    for T in sweep_thresholds:
        # For failures: rescue happens if base_margin <= T AND rescued_lr == 1
        rescued_T = sum((df_rescue_summary["base_margin"] <= T) & (df_rescue_summary["rescued_lr"] == 1))
        # For successes: broken happens if base_margin <= T AND is_broken == 1
        broken_T = sum((df_safety["base_margin"] <= T) & (df_safety["is_broken"] == 1))
        net_loc = rescued_T - broken_T

        print(f"T = {T:<8.3f} | {rescued_T:<20d} | {broken_T:<20d} | {net_loc:+15d}", flush=True)
        if net_loc > max_net:
            max_net = net_loc
            best_thresh = T

    print(f"\nBest Safe Margin Threshold: T = {best_thresh:.3f} (Rescued: {sum((df_rescue_summary['base_margin'] <= best_thresh) & (df_rescue_summary['rescued_lr'] == 1))}, Broken: {sum((df_safety['base_margin'] <= best_thresh) & (df_safety['is_broken'] == 1))})", flush=True)

    # Save summary tables
    df_rescue_summary.to_csv("FINAL_SUBMISSION/validation/rerank_v2_rescues_26.csv", index=False)
    df_safety.to_csv("FINAL_SUBMISSION/validation/rerank_v2_safety_76.csv", index=False)
    print("\nSaved forensic tables to 'FINAL_SUBMISSION/validation/rerank_v2_rescues_26.csv' and 'rerank_v2_safety_76.csv'.", flush=True)

if __name__ == "__main__":
    main()
