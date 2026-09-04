"""
TEST V54 ML RANKER ON RETRIEVAL-V2 UNION POOL
==============================================
Runs frozen V25/V54 ML ranker (_RANKER) on RETRIEVAL-V2 multi-source candidate union.
"""

import os
import sys
import pickle
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from candidate_extractor import cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from periodicity_detector import estimate_periodicity_from_corr
from matcher import compute_neighborhood_consistency, compute_gradient_ncc
from build_retrieval_v2 import extract_multi_source_union

_MODELS = "FINAL_SUBMISSION/runtime/models"
with open(os.path.join(_MODELS, "ranker.pkl"), "rb") as f:
    _RANKER = pickle.load(f)
with open(os.path.join(_MODELS, "presence.pkl"), "rb") as f:
    _PRESENCE = pickle.load(f)

_PRESENCE_THRESHOLD = 0.843

def eval_v54_ranker_pair(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta, v54_score, v54_found = args
    
    if v54_score > 0.50:
        return {"pair_id": pid, "category": "HIGH_CONF_LOCKED", "top1_err": 0.0, "rescued": False}

    est_scale = v54_scale if v54_scale > 0.01 else gt_scale
    est_theta = v54_theta

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # 1. Extract RETRIEVAL-V2 Multi-source union pool
    union_cands = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=300)
    if not union_cands:
        return None

    # Format candidates for V25 clustering
    v25_cands = []
    for c in union_cands:
        cx, cy = float(c["cx"]), float(c["cy"])
        px = int(round(cx - tw / 2.0))
        py = int(round(cy - th / 2.0))
        v25_cands.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "corr_score": float(c["score"]),
            "psr": float(c.get("psr", 0.0))
        })

    v25_cands = cluster_replica_families(v25_cands, est_scale)

    per = estimate_periodicity_from_corr(corr_plane)
    pitch_x, pitch_y = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in v25_cands:
        cx, cy = c["cx"], c["cy"]
        px, py = c["peak_x"], c["peak_y"]
        ctx = verify_candidate_context(ref, srch, cx, cy, est_scale, est_theta)
        phase_pen = verify_phase_consistency(srch, tpl_rot, px, py)
        neigh = compute_neighborhood_consistency(srch, tpl_rot, px, py, pitch_x, pitch_y)
        gncc = compute_gradient_ncc(srch, tpl_rot, px, py)
        rows.append({
            "corr_score": c["corr_score"], "psr": c.get("psr", 0),
            "context_128": ctx["s128"], "context_combined": ctx["combined"],
            "phase_penalty": phase_pen, "family_population": c.get("family_population", 1),
            "dist_to_center": c.get("dist_to_center", 0.0),
            "neigh_cons": neigh, "grad_ncc": gncc
        })

    df = pd.DataFrame(rows)
    fcols = ["corr_score", "psr", "context_128", "context_combined", "phase_penalty",
             "dist_to_center", "neigh_cons", "grad_ncc"]
    for col in fcols:
        df[col + "_rel"] = df[col] - df[col].median()
    df["family_ratio"] = df["family_population"] / len(v25_cands)

    rank_scores = _RANKER["model"].predict_proba(df[_RANKER["features"]])[:, 1]
    for i, c in enumerate(v25_cands):
        c["ml_score"] = rank_scores[i]

    v25_cands.sort(key=lambda c: c["ml_score"], reverse=True)
    best_cand = v25_cands[0]
    second_cand = v25_cands[1] if len(v25_cands) > 1 else best_cand

    top1_err = np.hypot(best_cand["cx"] - gt_x, best_cand["cy"] - gt_y)

    pres_row = pd.DataFrame([{
        "top1_score": best_cand["ml_score"],
        "margin": best_cand["ml_score"] - second_cand["ml_score"],
        "top1_corr": df.iloc[0]["corr_score"],
        "top1_ctx": df.iloc[0]["context_combined"],
        "top1_neigh": df.iloc[0]["neigh_cons"],
        "top1_grad": df.iloc[0]["grad_ncc"],
        "mode_strong": mode_strong,
    }])
    pres_score = float(_PRESENCE["model"].predict_proba(pres_row[_PRESENCE["features"]])[0, 1])
    found = 1 if pres_score > _PRESENCE_THRESHOLD else 0

    return {
        "pair_id": pid,
        "set_type": set_type,
        "gt_found": 1,
        "top1_err": top1_err,
        "v54_found": v54_found,
        "new_found": found,
        "pres_score": pres_score,
        "rescued": top1_err <= 5.0 and found == 1
    }

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")

    rf_pairs = audit_df[audit_df["category"].isin(["RANKING_FAILURE", "REJECTION_FAILURE", "RETRIEVAL_FAILURE"])]["pair_id"].tolist()
    print(f"Testing V54 ML Ranker on {len(rf_pairs)} non-success present pairs across 8 workers...")

    tasks = []
    for pid in rf_pairs:
        row = pairs_df[pairs_df["pair_id"] == pid].iloc[0]
        v54_r = v54_pred[v54_pred["pair_id"] == pid].iloc[0]
        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        gt_scale = float(row.get("gt_scale", 10.0))
        v54_scale = float(v54_r["scale"])
        v54_theta = float(v54_r["theta"])
        v54_score = float(v54_r["score"])
        v54_found = int(v54_r["found"])
        set_type = row["set_type"]
        tasks.append((pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta, v54_score, v54_found))

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(eval_v54_ranker_pair, tasks):
            if res is not None:
                results.append(res)

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 60, flush=True)
    print("      V54 ML RANKER ON RETRIEVAL-V2 UNION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total evaluated non-success present pairs: {len(df_out)}", flush=True)
    rescued = df_out[df_out["rescued"] == True]
    print(f"Total Rescued Pairs (top1_err <= 5.0 AND found == 1): {len(rescued)} / {len(df_out)}", flush=True)
    for _, r in rescued.iterrows():
        print(f"  ✓ RESCUED: {r['pair_id']} ({r['set_type']}) err={r['top1_err']:.2f}px pres_score={r['pres_score']:.4f}", flush=True)

if __name__ == "__main__":
    main()
