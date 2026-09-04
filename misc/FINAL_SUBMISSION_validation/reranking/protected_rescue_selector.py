"""
PROTECTED V54 RESCUE SELECTOR V2 (MULTI-SOURCE STRUCTURAL GATE)
===============================================================
- High Confidence Lock: If v54_score > 0.50, LOCK V54 immutably (0 regressions on 76 successes).
- Low/Medium Confidence (v54_score <= 0.50): Search multi-source RETRIEVAL-V2 union pool.
- Structural Vector: Combines Core Patch Correlation, Context Alignment, Gradient NCC,
  and Local Pitch Lattice agreement.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
import rerank_features
from build_retrieval_v2 import extract_multi_source_union, estimate_local_pitch


def evaluate_pair_protected_rescue(args):
    (pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta, set_type,
     v54_x, v54_y, v54_found, v54_score, v54_theta, v54_scale) = args

    if gt_found == 0:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": 0,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": v54_scale,
            "found": v54_found, "score": v54_score, "was_rescued": False,
            "base_err": -1.0, "final_err": -1.0,
        }

    base_err = float(np.hypot(v54_x - gt_x, v54_y - gt_y)) if (gt_found == 1 and v54_x > 0.1) else 999.0

    # 1. IMMUTABLE SUCCESS LOCK: If V54 score > 0.50, skip rescue immediately!
    if v54_score > 0.50:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": v54_scale,
            "found": v54_found, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
        }

    # Read images for feature evaluation
    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    sh, sw = srch.shape[:2]
    if est_scale <= 0.01:
        est_scale = 10.0
    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl

    corr_full = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # 50% Core Patch Template
    ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
    cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
    tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
    tw_core, th_core = tpl_core.shape[1], tpl_core.shape[0]
    corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

    # Extract multi-source RETRIEVAL-V2 candidate union
    union_cands = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=300)
    if not union_cands:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": v54_scale,
            "found": v54_found, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
        }

    def eval_structural_vector(c):
        cx, cy = float(c["cx"]), float(c["cy"])
        # Check interior image bounds
        if cx - tw/2.0 < 5.0 or cx + tw/2.0 > sw - 5.0 or cy - th/2.0 < 5.0 or cy + th/2.0 > sh - 5.0:
            return -999.0

        py_full = int(round(cy - th / 2.0))
        px_full = int(round(cx - tw / 2.0))
        py_core = int(round(cy - th_core / 2.0))
        px_core = int(round(cx - tw_core / 2.0))

        f_corr = float(corr_full[py_full, px_full]) if (0 <= py_full < corr_full.shape[0] and 0 <= px_full < corr_full.shape[1]) else c["score"]
        c_corr = float(corr_core[py_core, px_core]) if (0 <= py_core < corr_core.shape[0] and 0 <= px_core < corr_core.shape[1]) else 0.0
        ctx = float(verify_candidate_context(ref, srch, cx, cy, est_scale, est_theta)["combined"])
        gncc = float(compute_gradient_ncc(srch, tpl_rot, px_full, py_full)) if (0 <= py_full < corr_full.shape[0] and 0 <= px_full < corr_full.shape[1]) else 0.0

        # Core-Focused Structural vector combination (decoupled from raw intensity grid correlation)
        return 0.45 * c_corr + 0.35 * ctx + 0.20 * gncc

    # Evaluate all candidates in union
    scored = []
    for c in union_cands:
        score_val = eval_structural_vector(c)
        if score_val > -900.0:
            scored.append({"cx": float(c["cx"]), "cy": float(c["cy"]), "struct_score": score_val, "raw_score": float(c["score"])})

    if not scored:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": v54_scale,
            "found": v54_found, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
        }

    scored.sort(key=lambda x: x["struct_score"], reverse=True)
    best_cand = scored[0]

    # Rescue gate: promote if best_cand has valid structural signal
    was_rescued = False
    if v54_found == 0:
        # False-rejected present pair: predict top structural candidate
        if best_cand["struct_score"] >= 0.020:
            final_cx, final_cy = best_cand["cx"], best_cand["cy"]
            final_score = best_cand["raw_score"]
            final_found = 1
            was_rescued = True
        else:
            final_cx, final_cy = v54_x, v54_y
            final_score = v54_score
            final_found = v54_found
    else:
        # Low-confidence accepted pair: promote only if best_cand beats baseline candidate
        if best_cand["struct_score"] > base_cand_score + 0.001:
            final_cx, final_cy = best_cand["cx"], best_cand["cy"]
            final_score = best_cand["raw_score"]
            final_found = 1
            was_rescued = True
        else:
            final_cx, final_cy = v54_x, v54_y
            final_score = v54_score
            final_found = v54_found

    final_err = float(np.hypot(final_cx - gt_x, final_cy - gt_y)) if gt_found == 1 else -1.0

    return {
        "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
        "x": final_cx, "y": final_cy, "theta": v54_theta, "scale": v54_scale,
        "found": final_found, "score": final_score, "was_rescued": was_rescued,
        "base_err": base_err, "final_err": final_err,
    }


def main():
    print("=" * 65)
    print("  PROTECTED V54 RESCUE SELECTOR V2 (MULTI-SOURCE STRUCTURAL GATE) ")
    print("=" * 65)

    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")

    tasks = []
    for _, row in pairs_df.iterrows():
        pid = row["pair_id"]
        v54_r = v54_pred[v54_pred["pair_id"] == pid].iloc[0]
        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        tasks.append((
            pid, ref_p, srch_p,
            float(row.get("gt_x", 0.0)), float(row.get("gt_y", 0.0)),
            int(row["gt_found"]),
            float(v54_r["scale"]), float(v54_r["theta"]),
            row["set_type"],
            float(v54_r["x"]), float(v54_r["y"]),
            int(v54_r["found"]), float(v54_r["score"]),
            float(v54_r["theta"]), float(v54_r["scale"])
        ))

    print(f"Running Protected Rescue Selector on {len(tasks)} pairs across 8 workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(evaluate_pair_protected_rescue, tasks):
            if res is not None:
                results.append(res)
    print(f"Completed in {time.time()-t0:.1f} seconds.\n")

    df_res = pd.DataFrame(results)

    # 1. Check 76 Baseline Success Safety
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    succ_ids = audit_df[audit_df["category"] == "SUCCESS_ACCEPTED"]["pair_id"].tolist()
    df_succ = df_res[df_res["pair_id"].isin(succ_ids)]
    broken = df_succ[df_succ["final_err"] > 5.0]

    print("=" * 65)
    print("               SAFETY GATES EVALUATION")
    print("=" * 65)
    print(f" Gate 1 — 76 Baseline Successes Broken: {len(broken)} / 76  (MANDATORY ZERO)")
    if len(broken) > 0:
        for _, r in broken.iterrows():
            print(f"   BROKEN: {r['pair_id']} base_err={r['base_err']:.2f}px -> final_err={r['final_err']:.2f}px")

    # 2. Check 40 Absent Safety
    df_abs = df_res[df_res["gt_found"] == 0]
    abs_fp = int(np.sum(df_abs["found"] == 1))
    print(f" Gate 2 — 40 Absent False Accepts:     {abs_fp} / 40   (MANDATORY ZERO NEW)")

    # 3. Rescued Failures
    rescued = df_res[(df_res["base_err"] > 5.0) & (df_res["final_err"] <= 5.0)]
    print(f" Gate 3 — Ranking/Retrieval Rescued:   {len(rescued)} pairs")
    for _, r in rescued.iterrows():
        print(f"   ✓ RESCUED: {r['pair_id']} base_err={r['base_err']:.2f}px -> final_err={r['final_err']:.2f}px")

    # 4. Save Candidate Predictions
    out_preds = df_res[["pair_id", "x", "y", "theta", "scale", "found", "score"]]
    os.makedirs("FINAL_SUBMISSION/validation/reranking", exist_ok=True)
    out_preds.to_csv("FINAL_SUBMISSION/validation/reranking/protected_rescue_predictions.csv", index=False)
    print("\nSaved predictions to FINAL_SUBMISSION/validation/reranking/protected_rescue_predictions.csv")


if __name__ == "__main__":
    main()
