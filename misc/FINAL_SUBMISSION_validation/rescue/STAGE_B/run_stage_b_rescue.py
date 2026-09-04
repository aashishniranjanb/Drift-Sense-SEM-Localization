"""
STAGE-B PROTECTED V54 RESCUE SELECTOR
=====================================
Executes Stage B shadow rescue experiment across B1, B2, B3, B4.

DO NOT modify:
- FINAL_SUBMISSION_GOLDEN
- production register.py
- V54 rejection
- V54 calibration
- V54 pose
- Retrieval-V2 frozen artifacts
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
from build_retrieval_v2 import extract_multi_source_union, estimate_local_pitch, subpixel_peak_refine


def extract_13_structural_features(ref_img, search_img, cx, cy, est_scale, est_theta, corr_plane, tpl_rot):
    """Calculates 13 independent evidence features for a candidate (cx, cy)."""
    sh, sw = search_img.shape[:2]
    tw = max(16, int(round(ref_img.shape[1] / est_scale)))
    th = max(16, int(round(ref_img.shape[0] / est_scale)))

    if cx - tw/2.0 < 5.0 or cx + tw/2.0 > sw - 5.0 or cy - th/2.0 < 5.0 or cy + th/2.0 > sh - 5.0:
        return None

    px = int(round(cx - tw / 2.0))
    py = int(round(cy - th / 2.0))

    # 1. Intensity NCC
    f_ncc = float(corr_plane[py, px]) if (0 <= py < corr_plane.shape[0] and 0 <= px < corr_plane.shape[1]) else 0.0

    # 2. Gradient NCC
    f_grad = float(compute_gradient_ncc(search_img, tpl_rot, px, py)) if (0 <= py < corr_plane.shape[0] and 0 <= px < corr_plane.shape[1]) else 0.0

    # 3. Phase residual / penalty
    f_phase_pen = float(verify_phase_consistency(search_img, tpl_rot, px, py))

    # 4, 5, 6. Context 32, 64, 128
    ctx = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
    f_ctx32 = float(ctx["s32"])
    f_ctx64 = float(ctx["s64"])
    f_ctx128 = float(ctx["s128"])
    f_ctx_comb = float(ctx["combined"])

    # 7, 8, 9. Neighborhood, Row, and Column consistency
    lat = estimate_local_pitch(corr_plane, px, py)
    if lat is not None:
        pitch_x, pitch_y = lat["vx_x"], lat["vy_y"]
        f_neigh = float(compute_neighborhood_consistency(search_img, tpl_rot, px, py, pitch_x, pitch_y))
        f_row_cons = float(compute_neighborhood_consistency(search_img, tpl_rot, px, py, pitch_x, 0))
        f_col_cons = float(compute_neighborhood_consistency(search_img, tpl_rot, px, py, 0, pitch_y))
    else:
        f_neigh = f_row_cons = f_col_cons = 0.0

    # 10. Peak prominence (PSR)
    patch_size = 11
    y1, y2 = max(0, py - patch_size), min(corr_plane.shape[0], py + patch_size + 1)
    x1, x2 = max(0, px - patch_size), min(corr_plane.shape[1], px + patch_size + 1)
    local_patch = corr_plane[y1:y2, x1:x2]
    mean_val = float(np.mean(local_patch))
    std_val = float(np.std(local_patch))
    f_psr = (f_ncc - mean_val) / (std_val + 1e-5) if std_val > 1e-5 else 0.0

    # 11. Local curvature / sharpness
    if 1 <= py < corr_plane.shape[0] - 1 and 1 <= px < corr_plane.shape[1] - 1:
        laplacian = (4 * corr_plane[py, px] - corr_plane[py-1, px] - corr_plane[py+1, px] - corr_plane[py, px-1] - corr_plane[py, px+1])
        f_sharpness = float(laplacian)
    else:
        f_sharpness = 0.0

    # 12. Competitor distance (distance to center prior / bounds)
    f_dist_center = float(np.hypot(cx - sw / 2.0, cy - sh / 2.0))

    # 13. Lattice support
    f_lattice_sup = 1.0 if lat is not None else 0.0

    return {
        "ncc": f_ncc,
        "grad": f_grad,
        "phase_pen": f_phase_pen,
        "ctx32": f_ctx32,
        "ctx64": f_ctx64,
        "ctx128": f_ctx128,
        "ctx_comb": f_ctx_comb,
        "neigh": f_neigh,
        "row_cons": f_row_cons,
        "col_cons": f_col_cons,
        "psr": f_psr,
        "sharpness": f_sharpness,
        "dist_center": f_dist_center,
        "lattice_sup": f_lattice_sup,
    }


def evaluate_pair_stage_b(args):
    (pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta, set_type,
     v54_x, v54_y, v54_found, v54_score, v54_theta, v54_scale) = args

    base_err = float(np.hypot(v54_x - gt_x, v54_y - gt_y)) if (gt_found == 1 and v54_x > 0.1) else -1.0

    # =================================================================
    # B1: REJECTED-ONLY PROTECTED RESCUE RULE
    # If v54_found == 1: RETURN V54 IMMUTABLY (100% baseline preservation!)
    # =================================================================
    if v54_found == 1:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": v54_scale,
            "found": 1, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
            "gt_cand_rank": -1, "candidate_source": "v54_baseline",
            "rescue_confidence": 0.0, "v54_confidence": v54_score,
            "promotion_reason": "V54_ACCEPTED_LOCKED"
        }

    # Reads images for feature extraction
    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    sh, sw = srch.shape[:2]
    scale_eval = v54_scale if v54_scale > 0.01 else est_scale
    if scale_eval <= 0.01: scale_eval = 10.0

    tw = max(16, int(round(ref.shape[1] / scale_eval)))
    th = max(16, int(round(ref.shape[0] / scale_eval)))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, v54_theta) if abs(v54_theta) > 0.01 else tpl
    corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # Core template for illumination drift decoupling
    ch0, ch1 = int(round(th * 0.25)), int(round(th * 0.75))
    cw0, cw1 = int(round(tw * 0.25)), int(round(tw * 0.75))
    tpl_core = tpl_rot[ch0:ch1, cw0:cw1]
    corr_core = cv2.matchTemplate(srch.astype(np.float32), tpl_core, cv2.TM_CCOEFF_NORMED)

    # 1. Extract RETRIEVAL-V2 candidate pool (ranks 201-800)
    union_cands = extract_multi_source_union(ref, srch, scale_eval, v54_theta, max_total_k=800)
    if not union_cands:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": scale_eval,
            "found": 0, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
            "gt_cand_rank": -1, "candidate_source": "none",
            "rescue_confidence": 0.0, "v54_confidence": v54_score,
            "promotion_reason": "NO_CANDIDATES"
        }

    gt_rank = -1
    for idx, c in enumerate(union_cands):
        if gt_found == 1 and np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
            gt_rank = idx + 1
            break

    # Evaluate 13 structural features for candidates in pool
    scored_cands = []
    for c in union_cands:
        feats = extract_13_structural_features(ref, srch, c["cx"], c["cy"], scale_eval, v54_theta, corr_plane, tpl_rot)
        if feats is not None:
            # Core patch template score
            py_c = int(round(c["cy"] - tpl_core.shape[0] / 2.0))
            px_c = int(round(c["cx"] - tpl_core.shape[1] / 2.0))
            c_core = float(corr_core[py_c, px_c]) if (0 <= py_c < corr_core.shape[0] and 0 <= px_c < corr_core.shape[1]) else 0.0

            # Structural Vector Evidence Score (combines Core correlation, combined context, gradient NCC, and PSR)
            struct_score = (0.40 * c_core + 0.30 * feats["ctx_comb"] + 0.20 * feats["grad"] + 0.10 * min(1.0, feats["psr"] / 10.0)) - (0.15 * feats["phase_pen"])
            scored_cands.append({
                "cx": c["cx"], "cy": c["cy"],
                "struct_score": struct_score,
                "raw_score": c["score"],
                "source": c["source"],
                "feats": feats
            })

    if not scored_cands:
        return {
            "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
            "x": v54_x, "y": v54_y, "theta": v54_theta, "scale": scale_eval,
            "found": 0, "score": v54_score, "was_rescued": False,
            "base_err": base_err, "final_err": base_err,
            "gt_cand_rank": gt_rank, "candidate_source": "none",
            "rescue_confidence": 0.0, "v54_confidence": v54_score,
            "promotion_reason": "NO_VALID_INTERIOR_CANDIDATES"
        }

    scored_cands.sort(key=lambda x: x["struct_score"], reverse=True)
    best_cand = scored_cands[0]

    # Structural Gate for Absent Safety (B4): strict threshold requirement
    # Prevents false positive creation on 40 absent pairs
    RESCUE_STRICT_GATE = 0.650

    was_rescued = False
    if best_cand["struct_score"] >= RESCUE_STRICT_GATE:
        final_x, final_y = best_cand["cx"], best_cand["cy"]
        final_found = 1
        final_score = float(best_cand["struct_score"])
        was_rescued = True
        reason = "B1_STRUCTURAL_RESCUE_PROMOTED"
    else:
        final_x, final_y = v54_x, v54_y
        final_found = 0
        final_score = v54_score
        reason = "RESCUE_GATE_REJECTED"

    final_err = float(np.hypot(final_x - gt_x, final_y - gt_y)) if gt_found == 1 else -1.0

    return {
        "pair_id": pid, "set_type": set_type, "gt_found": gt_found,
        "x": final_x, "y": final_y, "theta": v54_theta, "scale": scale_eval,
        "found": final_found, "score": final_score, "was_rescued": was_rescued,
        "base_err": base_err, "final_err": final_err,
        "gt_cand_rank": gt_rank, "candidate_source": best_cand["source"],
        "rescue_confidence": float(best_cand["struct_score"]), "v54_confidence": v54_score,
        "promotion_reason": reason
    }


def main():
    print("=" * 65)
    print("  STAGE-B PROTECTED V54 RESCUE SELECTOR SHADOW AUDIT")
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

    print(f"Running Stage B Protected Selector across {len(tasks)} pairs (8 workers)...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(evaluate_pair_stage_b, tasks):
            if res is not None:
                results.append(res)
    print(f"Completed in {time.time()-t0:.1f} seconds.\n")

    df_res = pd.DataFrame(results)

    # 1. Baseline Success Safety (76 set)
    audit_df = pd.read_csv("FINAL_SUBMISSION/validation/candidate_pool_audit_140.csv")
    succ_ids = audit_df[audit_df["category"] == "SUCCESS_ACCEPTED"]["pair_id"].tolist()
    df_succ = df_res[df_res["pair_id"].isin(succ_ids)]
    broken = df_succ[df_succ["final_err"] > 5.0]

    # 2. Absent Safety (40 set)
    df_abs = df_res[df_res["gt_found"] == 0]
    base_abs_fp = 2  # V54 baseline has 2 absent false accepts (pair_060, pair_067)
    abs_fp = int(np.sum(df_abs["found"] == 1))
    new_abs_fp = max(0, abs_fp - base_abs_fp)

    # 3. Rescued Failures
    rescued = df_res[(df_res["base_err"] > 5.0) & (df_res["final_err"] <= 5.0)]

    print("=" * 65)
    print("           STAGE B EVALUATION & SAFETY GATES SUMMARY")
    print("=" * 65)
    print(f"  Gate 1 — Baseline Successes Broken: {len(broken)} / 76  (MUST BE 0)")
    print(f"  Gate 2 — New Absent False Accepts:   {new_abs_fp} / 40   (MUST BE 0)")
    print(f"  Gate 3 — Correct Rescue Recoveries: {len(rescued)} pairs")
    
    if len(rescued) > 0:
        print("\nSuccessfully Rescued Pairs:")
        for _, r in rescued.iterrows():
            print(f"  ✓ {r['pair_id']} ({r['set_type']}): base_err={r['base_err']:.2f}px -> final_err={r['final_err']:.2f}px (source={r['candidate_source']})")

    # Output directory setup
    out_dir = "FINAL_SUBMISSION/validation/rescue/STAGE_B"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Output Predictions
    df_preds = df_res[["pair_id", "x", "y", "theta", "scale", "found", "score"]]
    df_preds.to_csv(os.path.join(out_dir, "rescue_b1_predictions.csv"), index=False)

    # 2. Output Candidate Audit
    df_res.to_csv(os.path.join(out_dir, "rescue_candidate_audit.csv"), index=False)

    # 3. Output Report JSON
    report_json = {
        "baseline_score": 91.040,
        "baseline_successes_broken": len(broken),
        "new_absent_false_positives": new_abs_fp,
        "correct_rescues": len(rescued),
        "total_pairs_evaluated": len(df_res),
        "hard_promotion_gate_passed": bool(len(broken) == 0 and new_abs_fp == 0 and len(rescued) > 0)
    }
    with open(os.path.join(out_dir, "rescue_b1_report.json"), "w") as f:
        json.dump(report_json, f, indent=2)

    # 4. Output Report Markdown
    with open(os.path.join(out_dir, "rescue_b1_report.md"), "w") as f:
        f.write("# STAGE-B1 PROTECTED V54 RESCUE REPORT\n\n")
        f.write(f"- Baseline Successes Broken: **{len(broken)} / 76** (GATE REQUIREMENT: 0)\n")
        f.write(f"- New Absent False Positives: **{new_abs_fp} / 40** (GATE REQUIREMENT: 0)\n")
        f.write(f"- Correct Rescues Achieved: **{len(rescued)}**\n")
        f.write(f"- Hard Promotion Gate Passed: **{report_json['hard_promotion_gate_passed']}**\n")

    print(f"\nSaved all artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
