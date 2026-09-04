"""
INSPECT NEWLY RETRIEVED CANDIDATES STRUCTURAL EVIDENCE
======================================================
Analyzes the exact structural feature values and scores for the 14 newly retrieved candidates
to determine why they were rejected by the Stage B rescue gate.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from build_retrieval_v2 import extract_multi_source_union
from run_stage_b_rescue import extract_13_structural_features

def main():
    df_new = pd.read_csv("FINAL_SUBMISSION/validation/retrieval/retrieval_v2_new14.csv")
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")

    print("=" * 70)
    print("      FORENSIC INSPECTION OF 14 NEWLY RETRIEVED GT CANDIDATES")
    print("=" * 70)

    rows = []
    for _, new_row in df_new.iterrows():
        pid = new_row["pair_id"]
        row = pairs_df[pairs_df["pair_id"] == pid].iloc[0]
        v54_r = v54_pred[v54_pred["pair_id"] == pid].iloc[0]

        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        scale_eval = float(v54_r["scale"]) if float(v54_r["scale"]) > 0.01 else float(row.get("gt_scale", 10.0))
        theta_eval = float(v54_r["theta"])

        ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None: continue

        tw = max(16, int(round(ref.shape[1] / scale_eval)))
        th = max(16, int(round(ref.shape[0] / scale_eval)))
        tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        tpl_rot = rotate_image(tpl, theta_eval) if abs(theta_eval) > 0.01 else tpl
        corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

        ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
        cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
        tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
        corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

        cands = extract_multi_source_union(ref, srch, scale_eval, theta_eval, max_total_k=800)
        
        gt_cand = None
        gt_cand_rank = -1
        gt_cand_err = 999.0

        for idx, c in enumerate(cands):
            err = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
            if err <= 5.0 and gt_cand is None:
                gt_cand = c
                gt_cand_rank = idx + 1
                gt_cand_err = err

        if gt_cand is not None:
            feats = extract_13_structural_features(ref, srch, gt_cand["cx"], gt_cand["cy"], scale_eval, theta_eval, corr_plane, tpl_rot)
            py_c = int(round(gt_cand["cy"] - tpl_core.shape[0] / 2.0))
            px_c = int(round(gt_cand["cx"] - tpl_core.shape[1] / 2.0))
            c_core = float(corr_core[py_c, px_c]) if (0 <= py_c < corr_core.shape[0] and 0 <= px_c < corr_core.shape[1]) else 0.0
            
            struct_score = (0.40 * c_core + 0.30 * feats["ctx_comb"] + 0.20 * feats["grad"] + 0.10 * min(1.0, feats["psr"] / 10.0)) - (0.15 * feats["phase_pen"]) if feats else 0.0

            rows.append({
                "pair_id": pid,
                "set_type": new_row["set_type"],
                "gt_rank": gt_cand_rank,
                "gt_err": gt_cand_err,
                "gt_source": gt_cand["source"],
                "c_core": c_core,
                "ctx_comb": feats["ctx_comb"] if feats else 0,
                "grad": feats["grad"] if feats else 0,
                "phase_pen": feats["phase_pen"] if feats else 0,
                "struct_score": struct_score,
                "reason": "UNDER_0.65_GATE" if struct_score < 0.65 else "RANKED_BELOW_FALSE_REPLICA"
            })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv("FINAL_SUBMISSION/validation/rescue/STAGE_B/inspect_14_new_retrievals.csv", index=False)
    print("\nSaved forensic report to FINAL_SUBMISSION/validation/rescue/STAGE_B/inspect_14_new_retrievals.csv")

if __name__ == "__main__":
    main()
