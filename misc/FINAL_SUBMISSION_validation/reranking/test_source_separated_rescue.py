"""
TEST SOURCE-SEPARATED CANDIDATE RANKING & RESCUE
================================================
Evaluates GT ranking accuracy across each RETRIEVAL-V2 candidate source individually.
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
from build_retrieval_v2 import estimate_local_pitch

_MODELS = "FINAL_SUBMISSION/runtime/models"
with open(os.path.join(_MODELS, "ranker.pkl"), "rb") as f:
    _RANKER = pickle.load(f)

def test_pair_source_breakdown(args):
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
    corr = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # Core template
    ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
    cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
    tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
    corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

    # Extract source 1: Standard V25 intensity
    cands_v25 = extract_candidates_akhilesh(corr, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)
    
    # Extract source 2: Core patch correlation
    cands_core = extract_candidates_akhilesh(corr_core, tpl_core.shape[1], tpl_core.shape[0], ref, srch, est_scale, est_theta, max_final_k=200)

    def check_pool(cands):
        min_err = 999.0
        for c in cands:
            err = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
            if err < min_err:
                min_err = err
        return min_err

    err_v25 = check_pool(cands_v25)
    err_core = check_pool(cands_core)

    return {
        "pair_id": pid,
        "set_type": set_type,
        "v25_min_err": err_v25,
        "core_min_err": err_core,
        "core_recovered": (err_v25 > 5.0) and (err_core <= 5.0)
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
        for res in executor.map(test_pair_source_breakdown, tasks):
            if res is not None:
                results.append(res)

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 60, flush=True)
    print("      SOURCE-SEPARATED RECOVERY SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total non-success present pairs: {len(df_out)}", flush=True)
    print(f"V25 Pool Recall (err <= 5.0): {(df_out['v25_min_err'] <= 5.0).sum()} / {len(df_out)}", flush=True)
    print(f"Core Pool Recall (err <= 5.0): {(df_out['core_min_err'] <= 5.0).sum()} / {len(df_out)}", flush=True)
    print(f"Core Pool Recoveries over V25: {df_out['core_recovered'].sum()} / {len(df_out)}", flush=True)

    rec = df_out[df_out["core_recovered"] == True]
    if len(rec) > 0:
        for _, r in rec.iterrows():
            print(f"  ✓ CORE RECOVERED: {r['pair_id']} ({r['set_type']}) v25_err={r['v25_min_err']:.2f}px -> core_err={r['core_min_err']:.2f}px", flush=True)

if __name__ == "__main__":
    main()
