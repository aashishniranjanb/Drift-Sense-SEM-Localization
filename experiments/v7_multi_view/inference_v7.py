"""
Drift-Sense++ V7: Redundant Multi-View Retrieval + Confidence-Adaptive Registration Engine
Combines:
  1. 4 Complementary Representations (Intensity, Gradient, Orientation, High-Pass)
  2. 4 Local Sub-Template Structural Anchor Views
  3. Candidate UNION + Multi-View Voting -> Top-30 spatial candidates
  4. Strict Confidence Gate (Delta-S >= 0.010 & PSR >= 5.5) locking FFT peak
  5. PACE Residual Neural Ranker on ambiguous periodic candidates
  6. Dual Subpixel Estimator Consensus (Phase Correlation + 2D Paraboloid Fit)
"""

import os
import sys
import time
import numpy as np
import cv2
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pace_model import ProcessAwareContextEncoder
from generate_pace_dataset import extract_directional_overlaps, extract_patch_safe, normalize_intensity
from inference_car import compute_psr_and_sidelobe, evaluate_estimator_consensus, load_pace_model
from experiments.v7_multi_view.retrieval_v7 import redundant_multi_view_retrieval


def perform_v7_localization(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    model_path: str = None,
    use_anchors: bool = True,
    delta_s_threshold: float = 0.008,
    psr_threshold: float = 5.5,
    lambda_correction: float = 0.08,
) -> tuple[float, float, dict]:
    t_start = time.perf_counter()

    # Preprocess RGB to Grayscale
    if len(ref_img.shape) == 3 and ref_img.shape[2] in (3, 4):
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY if ref_img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if len(search_img.shape) == 3 and search_img.shape[2] in (3, 4):
        search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY if search_img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    # V7 Redundant Multi-View Retrieval -> Top-30 Candidates
    candidates, ret_meta = redundant_multi_view_retrieval(search_img, ref_img, use_anchors=use_anchors, k_top=30)

    if len(candidates) == 0:
        return search_cx, search_cy, {"status": "NO_CANDIDATES", "latency_ms": 0}

    # Evaluate Confidence Signals on Top-1 Candidate
    top1 = candidates[0]
    top2_score = candidates[1]["v7_rank_score"] if len(candidates) > 1 else 0.0
    delta_s = top1["v7_rank_score"] - top2_score
    c_i = top1["corr_plane"]
    psr, mean_side, std_side = compute_psr_and_sidelobe(c_i, top1["peak_x"], top1["peak_y"])

    # Strict Confidence Safety Gate
    is_high_confidence = (delta_s >= delta_s_threshold and psr >= psr_threshold) or (top1["ncc_score"] >= 0.85)

    if is_high_confidence:
        final_x, final_y, consensus_D, is_confident = evaluate_estimator_consensus(ref_100, search_img, top1)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return final_x, final_y, {
            "x": round(final_x, 2), "y": round(final_y, 2),
            "ncc_score": round(top1["ncc_score"], 4),
            "delta_s": round(delta_s, 4),
            "psr": round(psr, 2),
            "consensus_D_px": round(consensus_D, 2),
            "is_confident": is_confident,
            "path": "FAST_TRUSTED_FFT",
            "pace_activated": False,
            "latency_ms": round(elapsed_ms, 2),
            "status": "OK"
        }

    # Ambiguous: Activate PACE Residual Ranking
    model, device = load_pace_model(model_path)
    use_pace = (model is not None)

    if use_pace:
        ref_64 = normalize_intensity(cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA))
        ref_128 = normalize_intensity(cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA))
        ref_ovl = extract_directional_overlaps(ref_img, 500, 500, 32, offset=200)

        ref_64_t = torch.from_numpy(ref_64).unsqueeze(0).unsqueeze(0).to(device)
        ref_128_t = torch.from_numpy(ref_128).unsqueeze(0).unsqueeze(0).to(device)
        ref_ovl_t = torch.from_numpy(ref_ovl).unsqueeze(0).to(device)

        cand_64_list, cand_128_list, cand_ovl_list, cand_ncc_list = [], [], [], []
        # Evaluate Top-20 Candidates in PACE
        pace_eval_cands = candidates[:20]
        for c in pace_eval_cands:
            p64 = normalize_intensity(extract_patch_safe(search_img, c["cx"], c["cy"], 64))
            p128 = normalize_intensity(extract_patch_safe(search_img, c["cx"], c["cy"], 128))
            povl = extract_directional_overlaps(search_img, c["cx"], c["cy"], 32, offset=40)

            cand_64_list.append(torch.from_numpy(p64).unsqueeze(0))
            cand_128_list.append(torch.from_numpy(p128).unsqueeze(0))
            cand_ovl_list.append(torch.from_numpy(povl))
            cand_ncc_list.append(c["ncc_score"])

        cand_64_batch = torch.stack(cand_64_list).to(device)
        cand_128_batch = torch.stack(cand_128_list).to(device)
        cand_ovl_batch = torch.stack(cand_ovl_list).to(device)
        cand_ncc_batch = torch.tensor(cand_ncc_list, dtype=torch.float32).to(device)

        with torch.no_grad():
            z_ref = model.forward_encoder(ref_64_t, ref_128_t, ref_ovl_t)
            z_cands = model.forward_encoder(cand_64_batch, cand_128_batch, cand_ovl_batch)
            scores = model(z_ref, z_cands, cand_ncc_batch).cpu().numpy()[0]

        for i, c in enumerate(pace_eval_cands):
            pace_raw = float(scores[i])
            c["pace_score"] = pace_raw
            c["final_score"] = c["ncc_score"] + lambda_correction * pace_raw

        pace_eval_cands.sort(key=lambda c: c["final_score"], reverse=True)
        best_candidate_pool = pace_eval_cands
    else:
        best_candidate_pool = candidates

    top3 = best_candidate_pool[:3]
    for c in top3:
        c["dist_to_center"] = float(np.hypot(c["cx"] - search_cx, c["cy"] - search_cy))

    # Periodicity Disambiguation
    best = top3[0]
    is_ambiguous = False

    if len(top3) >= 2:
        final_delta = top3[0].get("final_score", top3[0]["ncc_score"]) - top3[1].get("final_score", top3[1]["ncc_score"])
        cand_dist = np.hypot(top3[0]["cx"] - top3[1]["cx"], top3[0]["cy"] - top3[1]["cy"])

        if final_delta < 0.005 and 15.0 <= cand_dist <= 120.0:
            is_ambiguous = True
            ambiguity_pool = [c for c in top3 if (top3[0].get("final_score", top3[0]["ncc_score"]) - c.get("final_score", c["ncc_score"])) < 0.005]
            best = min(ambiguity_pool, key=lambda c: c["dist_to_center"])

    final_x, final_y, consensus_D, is_confident = evaluate_estimator_consensus(ref_100, search_img, best)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    meta = {
        "x": round(final_x, 2),
        "y": round(final_y, 2),
        "ncc_score": round(best["ncc_score"], 4),
        "pace_score": round(best.get("pace_score", 0.0), 4),
        "delta_s": round(delta_s, 4),
        "psr": round(psr, 2),
        "consensus_D_px": round(consensus_D, 2),
        "is_confident": is_confident,
        "is_ambiguous": is_ambiguous,
        "pace_activated": True,
        "latency_ms": round(elapsed_ms, 2),
        "path": "V7_MULTI_VIEW_PACE",
        "status": "OK",
    }

    return final_x, final_y, meta
