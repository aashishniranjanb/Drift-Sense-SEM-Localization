"""
SOURCE ATTRIBUTION AUDIT FOR RETRIEVAL-V2
=========================================
Identifies the newly retrieved GT candidates (+14) and computes exact source attribution mapping.
Outputs:
- retrieval_v2_new14.csv
- SOURCE_ATTRIBUTION.csv
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
from candidate_extractor import extract_candidates_akhilesh
from build_retrieval_v2 import extract_multi_source_union

def analyze_pair_attribution(args):
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

    # 1. Baseline V25 top-200 candidate pool
    v25_cands = extract_candidates_akhilesh(corr_intensity, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)
    v25_min_err = min([np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in v25_cands]) if v25_cands else 999.0
    v25_rank = 999
    for idx, c in enumerate(v25_cands):
        if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
            v25_rank = idx + 1
            break

    # 2. RETRIEVAL-V2 multi-source union candidate pool
    union_cands = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=800)
    
    v2_min_err = 999.0
    v2_rank = 999
    gt_source = "NONE"

    for idx, c in enumerate(union_cands):
        err = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
        if err < v2_min_err:
            v2_min_err = err
        if err <= 5.0 and v2_rank == 999:
            v2_rank = idx + 1
            gt_source = c["source"]

    is_new = (v25_min_err > 5.0) and (v2_min_err <= 5.0)

    return {
        "pair_id": pid,
        "set_type": set_type,
        "gt_found": 1,
        "GT_x": gt_x,
        "GT_y": gt_y,
        "V54_rank": v25_rank,
        "V54_error": float(v25_min_err),
        "V2_rank": v2_rank,
        "V2_error": float(v2_min_err),
        "source_of_GT": gt_source,
        "is_newly_retrieved": is_new
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

    print(f"Running Source Attribution Audit across {len(tasks)} pairs (8 workers)...")
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(analyze_pair_attribution, tasks):
            if res is not None:
                results.append(res)

    df_res = pd.DataFrame(results)
    
    # Save newly retrieved 14 cases
    df_new14 = df_res[df_res["is_newly_retrieved"] == True].copy()
    os.makedirs("FINAL_SUBMISSION/validation/retrieval", exist_ok=True)
    df_new14.to_csv("FINAL_SUBMISSION/validation/retrieval/retrieval_v2_new14.csv", index=False)
    
    print("\n" + "=" * 65)
    print(f"     SOURCE ATTRIBUTION FOR {len(df_new14)} NEWLY RETRIEVED GT CANDIDATES")
    print("=" * 65)
    print(df_new14[["pair_id", "set_type", "V54_error", "V2_error", "V2_rank", "source_of_GT"]].to_string(index=False))

    # Source Summary Table
    source_counts = df_new14["source_of_GT"].value_counts().reset_index()
    source_counts.columns = ["Source_Combination", "Count"]
    source_counts.to_csv("FINAL_SUBMISSION/validation/retrieval/SOURCE_ATTRIBUTION.csv", index=False)

    print("\n" + "=" * 65)
    print("            SOURCE COMBINATION FREQUENCY SUMMARY")
    print("=" * 65)
    print(source_counts.to_string(index=False))

if __name__ == "__main__":
    main()
