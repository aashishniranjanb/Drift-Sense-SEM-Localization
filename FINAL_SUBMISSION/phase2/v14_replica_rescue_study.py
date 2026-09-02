import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append("phase2")
sys.path.append("fallbacks")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search
from pose_refinement import refine_pose
from inference_phase2 import (
    verify_candidate_context,
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation,
    cluster_replica_families,
    compute_spatial_fingerprint,
    rank_candidates,
    compute_ambiguity_index,
    rerank_with_pace,
    extract_presence_features,
    classify_presence
)

def extract_candidates_k(corr_plane, tw, th, max_k=100, r=5):
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    cands = []
    for rank in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.01 or np.isnan(max_val): break
        px, py = max_loc
        cands.append({
            "peak_x": px,
            "peak_y": py,
            "cx": px + tw / 2.0,
            "cy": py + th / 2.0,
            "corr_score": float(max_val)
        })
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return cands

def evaluate_variant(variant_name, pairs_df, max_k=100):
    start_time = time.time()
    results = []
    
    gt_retrieved_count = 0
    gt_top1_count = 0
    total_present = 0
    
    for idx, row in pairs_df.iterrows():
        pair_id = row["pair_id"]
        gt_found = int(row["gt_found"])
        set_type = row.get("set_type", "SetA" if gt_found == 1 else "SetC")
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        gt_scale = float(row.get("gt_scale", 10.0))
        gt_theta = float(row.get("gt_theta", 0.0))
        
        if gt_found == 1:
            total_present += 1
            
        ref_img = cv2.imread(os.path.join("data/phase2_dev", row["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join("data/phase2_dev", row["search_path"]), cv2.IMREAD_GRAYSCALE)
        sh, sw = search_img.shape[:2]
        
        # 1. Pose estimation
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
        
        est_scale = float(scale_res["best_scale"])
        est_theta = float(rot_res["best_theta"])
        rotated_tpl = rot_res["rotated_template"]
        th, tw = rotated_tpl.shape[:2]
        corr_plane = rot_res["corr_plane"]
        
        # 2. Extract Top-K candidates
        raw_cands = extract_candidates_k(corr_plane, tw, th, max_k=max_k, r=5)
        
        # Check if GT is in retrieved candidates
        gt_in_candidates = False
        if gt_found == 1:
            for c in raw_cands:
                if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
                    gt_in_candidates = True
                    break
            if gt_in_candidates:
                gt_retrieved_count += 1
                
        # 3. Enrich candidates
        enriched = []
        for c in raw_cands:
            px, py = c["peak_x"], c["peak_y"]
            cx, cy = c["cx"], c["cy"]
            y1, y2 = max(0, int(py)), min(sh, int(py + th))
            x1, x2 = max(0, int(px)), min(sw, int(px + tw))
            search_crop = search_img[y1:y2, x1:x2]
            
            psr, _, _ = compute_psr(corr_plane, px, py)
            ctx_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
            
            phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
            if search_crop.shape == (th, tw):
                phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_tpl, search_crop)
            phase_penalty = verify_phase_consistency(search_img, rotated_tpl, px, py)
            dist_to_center = np.hypot(cx - sw/2.0, cy - sh/2.0)
            
            enriched.append({
                "peak_x": px, "peak_y": py, "cx": cx, "cy": cy,
                "corr_score": c["corr_score"], "psr": psr,
                "context_32": ctx_res["s32"], "context_64": ctx_res["s64"],
                "context_128": ctx_res["s128"], "context_score": ctx_res["combined"],
                "phase_dx": phase_dx, "phase_dy": phase_dy,
                "phase_residual": phase_residual, "phase_penalty": phase_penalty,
                "center_prior": dist_to_center,
                "score_combined": float(0.50 * c["corr_score"] + 0.50 * ctx_res["combined"] - phase_penalty),
                "pace_score": 0.0
            })
            
        for i in range(len(enriched)):
            next_score = enriched[i+1]["corr_score"] if i+1 < len(enriched) else 0.0
            enriched[i]["peak_margin"] = enriched[i]["corr_score"] - next_score

        # 4. Apply Ranking Variant
        ranked = enriched
        if variant_name == "R0_CAR_Baseline":
            # Standard CAR
            ranked = cluster_replica_families(ranked, est_scale)
            for c in ranked:
                fam_members = [m for m in ranked if m.get("family_id") == c.get("family_id")]
                fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
                c.update(fp)
            ranked = rank_candidates(ranked)
            ambiguity_score, is_ambiguous = compute_ambiguity_index(ranked, est_scale)
            if is_ambiguous and len(ranked) > 0:
                ranked = rerank_with_pace(ref_img, search_img, ranked, est_scale, est_theta)
                for cand in ranked:
                    cand["rank_score"] = cand.get("rank_score", 0.0) - 0.08 * (cand["center_prior"] / (sw / 2.0))
                ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
            else:
                ranked.sort(key=lambda x: x.get("score_combined", 0.0), reverse=True)

        elif variant_name == "R1_Context128_Phase":
            # Emphasize Context-128 and Phase Residual directly
            for c in ranked:
                c["rank_score"] = float(0.35 * c["corr_score"] + 0.45 * c["context_128"] + 0.20 * c["context_64"] - 0.15 * c["phase_residual"] - c["phase_penalty"])
            ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)

        elif variant_name == "R2_Spatial_Family":
            ranked = cluster_replica_families(ranked, est_scale)
            for c in ranked:
                fam_members = [m for m in ranked if m.get("family_id") == c.get("family_id")]
                fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
                c.update(fp)
                c["rank_score"] = float(0.40 * c["corr_score"] + 0.30 * c["context_score"] + 0.20 * c.get("structural_integrity", 0.5) - 0.05 * (c["center_prior"] / (sw/2.0)))
            ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)

        elif variant_name == "R3_Full_Ensemble":
            ranked = cluster_replica_families(ranked, est_scale)
            for c in ranked:
                fam_members = [m for m in ranked if m.get("family_id") == c.get("family_id")]
                fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
                c.update(fp)
            ranked = rank_candidates(ranked)
            ambiguity_score, is_ambiguous = compute_ambiguity_index(ranked, est_scale)
            
            # Weighted contextual multi-scale fusion
            for cand in ranked:
                base_score = cand.get("rank_score", cand.get("score_combined", 0.0))
                ctx128_bonus = 0.25 * cand["context_128"]
                phase_res_penalty = 0.10 * cand["phase_residual"]
                center_penalty = 0.05 * (cand["center_prior"] / (sw / 2.0))
                cand["rank_score"] = float(0.65 * base_score + ctx128_bonus - phase_res_penalty - center_penalty)
                
            ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)

        # Best Candidate Selection
        if len(ranked) > 0:
            best_cand = ranked[0]
            rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta, best_cand["peak_x"], best_cand["peak_y"], corr_plane)
            best_cand["cx"] = rx
            best_cand["cy"] = ry
        else:
            best_cand = None

        # Check Conditional Top-1
        if gt_found == 1 and gt_in_candidates and best_cand is not None:
            if np.hypot(best_cand["cx"] - gt_x, best_cand["cy"] - gt_y) <= 5.0:
                gt_top1_count += 1

        # Presence Classification
        if best_cand is not None:
            ctx_score = best_cand["context_score"]
            phase_res = best_cand["phase_residual"]
            px, py = best_cand["peak_x"], best_cand["peak_y"]
        else:
            ctx_score = 0.0
            phase_res = 0.0
            px, py = 0, 0
            
        p_feats = extract_presence_features(corr_plane, px, py, rotated_tpl, search_img, context_score=ctx_score, phase_residual=phase_res)
        found, raw_p_score = classify_presence(p_feats)
        conf_score = raw_p_score if found == 1 else (1.0 - raw_p_score)
        
        # Localization Error
        loc_err = -1.0
        if gt_found == 1 and found == 1 and best_cand is not None:
            loc_err = float(np.hypot(best_cand["cx"] - gt_x, best_cand["cy"] - gt_y))
            
        results.append({
            "pair_id": pair_id,
            "set_type": set_type,
            "gt_found": gt_found,
            "found": found,
            "loc_err": loc_err,
            "score": conf_score
        })
        
    total_time = time.time() - start_time
    avg_latency = total_time / len(pairs_df)
    
    df_res = pd.DataFrame(results)
    
    # Calculate Set A and Set B <= 5px
    setA = df_res[df_res["set_type"] == "SetA"]
    setB = df_res[df_res["set_type"] == "SetB"]
    
    setA_le5 = np.mean((setA["found"] == 1) & (setA["loc_err"] >= 0) & (setA["loc_err"] <= 5.0)) * 100.0 if len(setA) > 0 else 0.0
    setB_le5 = np.mean((setB["found"] == 1) & (setB["loc_err"] >= 0) & (setB["loc_err"] <= 5.0)) * 100.0 if len(setB) > 0 else 0.0
    weighted_loc = 0.45 * setA_le5 + 0.55 * setB_le5
    
    # Rejection F1
    tp_rej = np.sum((df_res["gt_found"] == 0) & (df_res["found"] == 0))
    fp_rej = np.sum((df_res["gt_found"] == 1) & (df_res["found"] == 0))
    fn_rej = np.sum((df_res["gt_found"] == 0) & (df_res["found"] == 1))
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    
    retrieval_pct = (gt_retrieved_count / total_present) * 100.0
    cond_top1_pct = (gt_top1_count / gt_retrieved_count * 100.0) if gt_retrieved_count > 0 else 0.0
    
    return {
        "variant": variant_name,
        "max_k": max_k,
        "retrieval_recall": retrieval_pct,
        "cond_top1_accuracy": cond_top1_pct,
        "weighted_loc": weighted_loc,
        "setA_le5": setA_le5,
        "setB_le5": setB_le5,
        "f1_rejection": f1_rej,
        "latency_sec": avg_latency
    }

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    variants = [
        ("R0_CAR_Baseline", 50),
        ("R0_CAR_Baseline", 100),
        ("R1_Context128_Phase", 100),
        ("R2_Spatial_Family", 100),
        ("R3_Full_Ensemble", 100),
        ("R3_Full_Ensemble", 200)
    ]
    
    print("Running V14 Replica Rescue Ablations on 180 Dev Cases...")
    records = []
    for vname, k in variants:
        res = evaluate_variant(vname, pairs_df, max_k=k)
        records.append(res)
        print(f"[{vname} (K={k})] -> Retrieval: {res['retrieval_recall']:.2f}% | Cond Top-1: {res['cond_top1_accuracy']:.2f}% | Weighted Loc: {res['weighted_loc']:.2f}% (SetA: {res['setA_le5']:.2f}%, SetB: {res['setB_le5']:.2f}%) | Latency: {res['latency_sec']:.2f}s")
        
    df_out = pd.DataFrame(records)
    df_out.to_csv("results/v14/replica_ablation.csv", index=False)
    print("Replica ablation study saved to results/v14/replica_ablation.csv")

if __name__ == "__main__":
    main()
