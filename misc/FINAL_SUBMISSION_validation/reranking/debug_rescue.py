"""
PARALLEL DEBUG STRUCTURAL CANDIDATE RESCUE
==========================================
Analyzes structural vector scores across all 64 non-success present pairs in parallel.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from context_matcher import verify_candidate_context
from matcher import compute_gradient_ncc
from build_retrieval_v2 import extract_multi_source_union

def debug_pair(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta = args
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

    ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
    cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
    tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
    tw_core, th_core = tpl_core.shape[1], tpl_core.shape[0]
    corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

    cands = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=300)
    
    scored = []
    gt_cand_info = None
    gt_cand_err = 999.0

    for c in cands:
        cx, cy = float(c["cx"]), float(c["cy"])
        if cx - tw/2.0 < 5.0 or cx + tw/2.0 > sw - 5.0 or cy - th/2.0 < 5.0 or cy + th/2.0 > sh - 5.0:
            continue

        py_full = int(round(cy - th / 2.0))
        px_full = int(round(cx - tw / 2.0))
        py_core = int(round(cy - th_core / 2.0))
        px_core = int(round(cx - tw_core / 2.0))

        f_corr = float(corr_full[py_full, px_full]) if (0 <= py_full < corr_full.shape[0] and 0 <= px_full < corr_full.shape[1]) else 0.0
        c_corr = float(corr_core[py_core, px_core]) if (0 <= py_core < corr_core.shape[0] and 0 <= px_core < corr_core.shape[1]) else 0.0
        ctx = float(verify_candidate_context(ref, srch, cx, cy, est_scale, est_theta)["combined"])
        gncc = float(compute_gradient_ncc(srch, tpl_rot, px_full, py_full)) if (0 <= py_full < corr_full.shape[0] and 0 <= px_full < corr_full.shape[1]) else 0.0

        score_val = 0.45 * c_corr + 0.35 * ctx + 0.20 * gncc
        err = np.hypot(cx - gt_x, cy - gt_y)

        info = {
            "cx": cx, "cy": cy, "err": err, "score_val": score_val,
            "c_corr": c_corr, "ctx": ctx, "gncc": gncc, "f_corr": f_corr,
            "source": c.get("source", "unknown")
        }
        scored.append(info)
        if err < gt_cand_err:
            gt_cand_err = err
            gt_cand_info = info

    if not scored:
        return None

    scored.sort(key=lambda x: x["score_val"], reverse=True)
    top1 = scored[0]

    gt_rank = None
    for idx, item in enumerate(scored):
        if item["err"] <= 5.0:
            gt_rank = idx + 1
            break

    return {
        "pair_id": pid,
        "set_type": set_type,
        "gt_in_pool": gt_cand_err <= 5.0,
        "gt_min_err": gt_cand_err,
        "gt_rank": gt_rank,
        "top1_err": top1["err"],
        "top1_score": top1["score_val"],
        "gt_score": gt_cand_info["score_val"] if gt_cand_info else 0.0,
    }

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")

    rf_pairs = audit_df[audit_df["category"].isin(["RANKING_FAILURE", "REJECTION_FAILURE", "RETRIEVAL_FAILURE"])]["pair_id"].tolist()
    print(f"Analyzing {len(rf_pairs)} non-success present pairs across 8 workers...")

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
        set_type = row["set_type"]
        tasks.append((pid, ref_p, srch_p, gt_x, gt_y, gt_scale, set_type, v54_scale, v54_theta))

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(debug_pair, tasks):
            if res is not None:
                results.append(res)

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 60, flush=True)
    print("           DEBUG STRUCTURAL RESCUE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total evaluated non-success present pairs: {len(df_out)}", flush=True)
    print(f"GT present in RETRIEVAL-V2 pool (err <= 5.0): {df_out['gt_in_pool'].sum()} / {len(df_out)}", flush=True)
    print(f"GT ranked #1 by structural vector: {(df_out['top1_err'] <= 5.0).sum()} / {len(df_out)}", flush=True)

    gt_top1 = df_out[df_out["top1_err"] <= 5.0]
    if len(gt_top1) > 0:
        print("\nPairs where structural vector correctly ranks GT #1:", flush=True)
        for _, row in gt_top1.iterrows():
            print(f"  {row['pair_id']} ({row['set_type']}): top1_err={row['top1_err']:.2f}px top1_score={row['top1_score']:.4f} (gt_rank={row['gt_rank']})", flush=True)

    # Breakdown by category
    print("\nDetailed Breakdown of GT Rank across 64 failures:", flush=True)
    print(df_out["gt_rank"].value_counts().sort_index(), flush=True)

if __name__ == "__main__":
    main()
