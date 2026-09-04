"""
EFFECTIVE 131 AUDIT FOR RETRIEVAL-V2
===================================
Classifies all 140 present pairs into:
- DIRECT: candidate in pool <= 5.0px
- NEAR: candidate in pool 5.0px - 10.0px
- REFINABLE: subpixel parabolic fit error <= 1.0px
- NOT_REFINABLE: error > 1.0px after refinement
Outputs:
- effective_131_audit.csv
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
from build_retrieval_v2 import extract_multi_source_union, subpixel_peak_refine

def audit_effective_pair(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta, set_type = args
    if gt_found == 0:
        return None

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    tw = max(16, int(round(ref.shape[1] / est_scale)))
    th = max(16, int(round(ref.shape[0] / est_scale)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr_intensity = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    union_cands = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=800)
    
    nearest_raw_err = 999.0
    nearest_refined_err = 999.0
    nearest_cand = None

    for c in union_cands:
        cx, cy = float(c["cx"]), float(c["cy"])
        raw_err = np.hypot(cx - gt_x, cy - gt_y)
        if raw_err < nearest_raw_err:
            nearest_raw_err = raw_err
            nearest_cand = c

    if nearest_cand is not None:
        px, py = nearest_cand["peak_x"], nearest_cand["peak_y"]
        sp_x, sp_y = subpixel_peak_refine(corr_intensity, px, py)
        ref_cx = sp_x + tw / 2.0
        ref_cy = sp_y + th / 2.0
        nearest_refined_err = float(np.hypot(ref_cx - gt_x, ref_cy - gt_y))

    # Classification
    if nearest_raw_err <= 5.0:
        cat = "DIRECT"
    elif nearest_raw_err <= 10.0:
        cat = "NEAR"
    else:
        cat = "FAR"

    if nearest_refined_err <= 1.0:
        refine_cat = "REFINABLE_SUBPIXEL_1PX"
    elif nearest_refined_err <= 5.0:
        refine_cat = "REFINABLE_SUBPIXEL_5PX"
    else:
        refine_cat = "NOT_REFINABLE"

    return {
        "pair_id": pid,
        "set_type": set_type,
        "gt_found": 1,
        "GT_x": gt_x,
        "GT_y": gt_y,
        "nearest_cand_x": nearest_cand["cx"] if nearest_cand else 0.0,
        "nearest_cand_y": nearest_cand["cy"] if nearest_cand else 0.0,
        "raw_cand_dist_px": nearest_raw_err,
        "refined_dist_px": nearest_refined_err,
        "dist_category": cat,
        "refinement_category": refine_cat
    }

def main():
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
            float(v54_r["scale"]) if float(v54_r["scale"]) > 0.01 else float(row.get("gt_scale", 10.0)),
            float(v54_r["theta"]),
            row["set_type"]
        ))

    print(f"Running Effective 131 Audit across {len(tasks)} pairs (8 workers)...")
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(audit_effective_pair, tasks):
            if res is not None:
                results.append(res)

    df_res = pd.DataFrame(results)
    os.makedirs("FINAL_SUBMISSION/validation/retrieval", exist_ok=True)
    df_res.to_csv("FINAL_SUBMISSION/validation/retrieval/effective_131_audit.csv", index=False)

    print("\n" + "=" * 65)
    print("      EFFECTIVE 131 AUDIT SUMMARY (140 PRESENT PAIRS)")
    print("=" * 65)
    print("1. Direct Candidate Distance Category Breakdown:")
    print(df_res["dist_category"].value_counts().to_string())

    print("\n2. Subpixel Refinement Category Breakdown:")
    print(df_res["refinement_category"].value_counts().to_string())

    print("\nSaved report to FINAL_SUBMISSION/validation/retrieval/effective_131_audit.csv")

if __name__ == "__main__":
    main()
