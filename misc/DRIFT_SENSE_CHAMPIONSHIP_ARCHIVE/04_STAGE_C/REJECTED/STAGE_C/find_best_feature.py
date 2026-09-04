import os
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
sys.path.insert(0, "FINAL_SUBMISSION/validation")

from utils import rotate_image
from build_retrieval_v2 import extract_multi_source_union
from context_matcher import verify_candidate_context
from matcher import compute_gradient_ncc
from phase_verifier import verify_phase_consistency

def extract_features(ref, srch, cx, cy, scale, theta, tpl_core, corr_plane, corr_core, tw, th, sh, sw):
    if cx - tw/2.0 < 5.0 or cx + tw/2.0 > sw - 5.0 or cy - th/2.0 < 5.0 or cy + th/2.0 > sh - 5.0:
        return None
    py_c = int(round(cy - tpl_core.shape[0] / 2.0))
    px_c = int(round(cx - tpl_core.shape[1] / 2.0))
    c_core = float(corr_core[py_c, px_c]) if (0 <= py_c < corr_core.shape[0] and 0 <= px_c < corr_core.shape[1]) else 0.0
    ctx = verify_candidate_context(ref, srch, cx, cy, scale, theta)
    px_full = int(round(cx - tw / 2.0))
    py_full = int(round(cy - th / 2.0))
    f_grad = float(compute_gradient_ncc(srch, rotate_image(cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA), theta) if abs(theta)>0.01 else cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA), px_full, py_full)) if (0 <= py_full < corr_plane.shape[0] and 0 <= px_full < corr_plane.shape[1]) else 0.0
    f_phase_pen = float(verify_phase_consistency(srch, rotate_image(cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA), theta) if abs(theta)>0.01 else cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA), px_full, py_full))
    struct_score = (0.40 * c_core + 0.35 * float(ctx["combined"]) + 0.25 * f_grad) - (0.15 * f_phase_pen)
    
    return {
        "c_core": c_core,
        "ctx_s32": float(ctx["s32"]),
        "ctx_s64": float(ctx["s64"]),
        "ctx_s128": float(ctx["s128"]),
        "ctx_comb": float(ctx["combined"]),
        "f_grad": f_grad,
        "f_phase_pen": f_phase_pen,
        "struct_score": struct_score
    }

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    
    ranking_failures = audit_df[audit_df["category"] == "RANKING_FAILURE"]["pair_id"].tolist()
    
    gt_top1 = {k: 0 for k in ["struct_score", "c_core", "ctx_s32", "ctx_s64", "ctx_s128", "ctx_comb", "f_grad"]}
    
    for pid in ranking_failures:
        row = pairs_df[pairs_df["pair_id"] == pid].iloc[0]
        v54_r = v54_pred[v54_pred["pair_id"] == pid].iloc[0]
        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        gt_scale = float(row.get("gt_scale", 10.0))
        
        v54_theta = float(v54_r["theta"])
        v54_scale = float(v54_r["scale"]) if float(v54_r["scale"]) > 0.01 else gt_scale
        
        ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
        sh, sw = srch.shape[:2]
        
        tw = max(16, int(round(ref.shape[1] / v54_scale)))
        th = max(16, int(round(ref.shape[0] / v54_scale)))
        tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        tpl_rot = rotate_image(tpl, v54_theta) if abs(v54_theta) > 0.01 else tpl
        corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)
        
        ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
        cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
        tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
        corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)
        
        cands = extract_multi_source_union(ref, srch, v54_scale, v54_theta, max_total_k=200)
        
        scored_cands = []
        for c in cands:
            cx, cy = c["cx"], c["cy"]
            feats = extract_features(ref, srch, cx, cy, v54_scale, v54_theta, tpl_core, corr_plane, corr_core, tw, th, sh, sw)
            if feats is not None:
                c_copy = c.copy()
                c_copy.update(feats)
                scored_cands.append(c_copy)
                
        if not scored_cands:
            continue
            
        gt_cand_match = min(scored_cands, key=lambda x: np.hypot(x["cx"] - gt_x, x["cy"] - gt_y))
        gt_dist = np.hypot(gt_cand_match["cx"] - gt_x, gt_cand_match["cy"] - gt_y)
        
        if gt_dist > max(25, tw*0.25):
            continue
            
        for feat in gt_top1.keys():
            scored_cands.sort(key=lambda x: x[feat], reverse=True)
            if scored_cands[0] == gt_cand_match:
                gt_top1[feat] += 1
                
    print("Features where GT is absolute #1 in the pool:")
    for k, v in gt_top1.items():
        print(f"{k}: {v} / {len(ranking_failures)}")

if __name__ == "__main__":
    main()
