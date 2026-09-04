import os
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
sys.path.insert(0, "FINAL_SUBMISSION/validation")

from utils import rotate_image
from build_retrieval_v2 import extract_multi_source_union, estimate_local_pitch, subpixel_peak_refine
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
    
    # Analyze RANKING_FAILURE
    ranking_failures = audit_df[audit_df["category"] == "RANKING_FAILURE"]["pair_id"].tolist()
    
    print(f"Analyzing {len(ranking_failures)} RANKING_FAILURES for Stage C pairwise disambiguation...")
    
    results = []
    
    for pid in ranking_failures:
        row = pairs_df[pairs_df["pair_id"] == pid].iloc[0]
        v54_r = v54_pred[v54_pred["pair_id"] == pid].iloc[0]
        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        gt_scale = float(row.get("gt_scale", 10.0))
        
        v54_x, v54_y = float(v54_r["x"]), float(v54_r["y"])
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
        
        cands = extract_multi_source_union(ref, srch, v54_scale, v54_theta, max_total_k=800)
        
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
            
        scored_cands.sort(key=lambda x: x["struct_score"], reverse=True)
        
        v54_cand_match = min(scored_cands, key=lambda x: np.hypot(x["cx"] - v54_x, x["cy"] - v54_y))
        v54_cand_feats = v54_cand_match
        
        gt_cand_match = min(scored_cands, key=lambda x: np.hypot(x["cx"] - gt_x, x["cy"] - gt_y))
        gt_dist = np.hypot(gt_cand_match["cx"] - gt_x, gt_cand_match["cy"] - gt_y)
        
        if gt_dist <= max(25, tw*0.25):
            res = {
                "pair_id": pid,
                "v54_cx": v54_cand_match["cx"],
                "v54_cy": v54_cand_match["cy"],
                "v54_struct": v54_cand_match["struct_score"],
                "gt_cx": gt_cand_match["cx"],
                "gt_cy": gt_cand_match["cy"],
                "gt_struct": gt_cand_match["struct_score"],
                "delta_struct": gt_cand_match["struct_score"] - v54_cand_match["struct_score"],
                "delta_c_core": gt_cand_match["c_core"] - v54_cand_match["c_core"],
                "delta_ctx_s32": gt_cand_match["ctx_s32"] - v54_cand_match["ctx_s32"],
                "delta_ctx_s64": gt_cand_match["ctx_s64"] - v54_cand_match["ctx_s64"],
                "delta_ctx_s128": gt_cand_match["ctx_s128"] - v54_cand_match["ctx_s128"],
                "delta_ctx_comb": gt_cand_match["ctx_comb"] - v54_cand_match["ctx_comb"],
                "delta_grad": gt_cand_match["f_grad"] - v54_cand_match["f_grad"],
                "delta_phase_pen": gt_cand_match["f_phase_pen"] - v54_cand_match["f_phase_pen"],
            }
            
            pitch = estimate_local_pitch(corr_plane, v54_cand_match["cx"] - tw/2.0, v54_cand_match["cy"] - th/2.0, search_radius=120)
            if pitch:
                vx_x, vx_y = pitch["vx_x"], pitch["vx_y"]
                vy_x, vy_y = pitch["vy_x"], pitch["vy_y"]
                dx = gt_cand_match["cx"] - v54_cand_match["cx"]
                dy = gt_cand_match["cy"] - v54_cand_match["cy"]
                
                denom = vx_x * vy_y - vx_y * vy_x
                if abs(denom) > 1e-5:
                    u = (dx * vy_y - dy * vy_x) / denom
                    v = (vx_x * dy - vx_y * dx) / denom
                    lat_dev = np.hypot(u - round(u), v - round(v))
                    res["lattice_dev"] = lat_dev
                else:
                    res["lattice_dev"] = -1.0
            else:
                res["lattice_dev"] = -1.0
                
            results.append(res)
            
    df_res = pd.DataFrame(results)
    os.makedirs("FINAL_SUBMISSION/validation/rescue/STAGE_C", exist_ok=True)
    df_res.to_csv("FINAL_SUBMISSION/validation/rescue/STAGE_C/pairwise_deltas.csv", index=False)
    
    print("\nPairwise Deltas (GT - V54) on RANKING_FAILURES:")
    if not df_res.empty:
        summary = df_res[["delta_struct", "delta_c_core", "delta_ctx_s32", "delta_ctx_s64", "delta_ctx_s128", "delta_grad", "delta_phase_pen"]].mean()
        print("Mean Deltas:")
        print(summary)
        print("\nCases where GT is better in specific features:")
        print(f"Struct Score:  {sum(df_res['delta_struct'] > 0)} / {len(df_res)}")
        print(f"Core NCC:      {sum(df_res['delta_c_core'] > 0)} / {len(df_res)}")
        print(f"Context 32:    {sum(df_res['delta_ctx_s32'] > 0)} / {len(df_res)}")
        print(f"Context 64:    {sum(df_res['delta_ctx_s64'] > 0)} / {len(df_res)}")
        print(f"Context 128:   {sum(df_res['delta_ctx_s128'] > 0)} / {len(df_res)}")
        print(f"Gradient:      {sum(df_res['delta_grad'] > 0)} / {len(df_res)}")
        print(f"Phase Pen:     {sum(df_res['delta_phase_pen'] < 0)} / {len(df_res)}") 
        print(f"On Lattice (<0.15): {sum(df_res['lattice_dev'] < 0.15)} / {len(df_res[df_res['lattice_dev'] != -1.0])}")
        
if __name__ == "__main__":
    main()
