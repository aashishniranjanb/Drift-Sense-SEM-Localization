"""
EVALUATE CORE CANDIDATE RECOVERY & RANKING
==========================================
Ranks Core template candidates using V25 ranker to check exact subpixel error and ML score.
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
from candidate_extractor import extract_candidates_akhilesh, cluster_replica_families
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from periodicity_detector import estimate_periodicity_from_corr
from matcher import compute_neighborhood_consistency, compute_gradient_ncc

_MODELS = "FINAL_SUBMISSION/runtime/models"
with open(os.path.join(_MODELS, "ranker.pkl"), "rb") as f:
    _RANKER = pickle.load(f)
with open(os.path.join(_MODELS, "presence.pkl"), "rb") as f:
    _PRESENCE = pickle.load(f)

_PRESENCE_THRESHOLD = 0.843

def eval_core_pair(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta, v54_score = args
    if v54_score > 0.50:
        return None

    est_scale = v54_scale if v54_scale > 0.01 else gt_scale
    est_theta = v54_theta

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    sh, sw = srch.shape[:2]
    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr_full = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # Core template
    ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
    cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
    tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
    corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

    # 1. V25 candidates
    cands_v25 = extract_candidates_akhilesh(corr_full, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)
    v25_min_err = min([np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in cands_v25]) if cands_v25 else 999.0

    # 2. Core candidates
    cands_core = extract_candidates_akhilesh(corr_core, tpl_core.shape[1], tpl_core.shape[0], ref, srch, est_scale, est_theta, max_final_k=200)
    if not cands_core:
        return None
    core_min_err = min([np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in cands_core])

    cands_core = cluster_replica_families(cands_core, est_scale)

    per = estimate_periodicity_from_corr(corr_full)
    pitch_x, pitch_y = per["pitch_x"], per["pitch_y"]
    mode_strong = 1 if per["mode"] == "STRONG" else 0

    rows = []
    for c in cands_core:
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
    df["family_ratio"] = df["family_population"] / len(cands_core)

    rank_scores = _RANKER["model"].predict_proba(df[_RANKER["features"]])[:, 1]
    for i, c in enumerate(cands_core):
        c["ml_score"] = rank_scores[i]

    cands_core.sort(key=lambda c: c["ml_score"], reverse=True)
    best_core = cands_core[0]
    second_core = cands_core[1] if len(cands_core) > 1 else best_core
    top1_core_err = np.hypot(best_core["cx"] - gt_x, best_core["cy"] - gt_y)

    pres_row = pd.DataFrame([{
        "top1_score": best_core["ml_score"],
        "margin": best_core["ml_score"] - second_core["ml_score"],
        "top1_corr": df.iloc[0]["corr_score"],
        "top1_ctx": df.iloc[0]["context_combined"],
        "top1_neigh": df.iloc[0]["neigh_cons"],
        "top1_grad": df.iloc[0]["grad_ncc"],
        "mode_strong": mode_strong,
    }])
    pres_score = float(_PRESENCE["model"].predict_proba(pres_row[_PRESENCE["features"]])[0, 1])

    return {
        "pair_id": pid,
        "set_type": set_type,
        "v25_min_err": v25_min_err,
        "core_min_err": core_min_err,
        "top1_core_err": top1_core_err,
        "pres_score": pres_score,
        "core_rescued": (v25_min_err > 5.0) and (top1_core_err <= 5.0) and (pres_score > 0.50)
    }

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")

    rf_pairs = audit_df[audit_df["category"].isin(["RANKING_FAILURE", "REJECTION_FAILURE", "RETRIEVAL_FAILURE"])]["pair_id"].tolist()

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
        set_type = row["set_type"]
        tasks.append((pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta, v54_score))

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(eval_core_pair, tasks):
            if res is not None:
                results.append(res)

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 60, flush=True)
    print("      EVALUATE CORE RESCUE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total non-success present pairs evaluated: {len(df_out)}", flush=True)
    print(f"Core Pool GT Present (err <= 5.0): {(df_out['core_min_err'] <= 5.0).sum()} / {len(df_out)}", flush=True)
    print(f"Core Ranker GT Ranked #1 (err <= 5.0): {(df_out['top1_core_err'] <= 5.0).sum()} / {len(df_out)}", flush=True)
    
    rec = df_out[df_out["top1_core_err"] <= 5.0]
    if len(rec) > 0:
        for _, r in rec.iterrows():
            print(f"  [CORE TOP-1 SUCCESS] {r['pair_id']} ({r['set_type']}) v25_err={r['v25_min_err']:.2f}px -> core_top1_err={r['top1_core_err']:.2f}px (pres_score={r['pres_score']:.4f})", flush=True)

if __name__ == "__main__":
    main()
