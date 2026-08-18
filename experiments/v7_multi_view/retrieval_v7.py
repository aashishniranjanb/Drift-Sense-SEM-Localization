"""
V7 Redundant Multi-View Retrieval Engine
Combines:
  1. Intensity FFT-NCC
  2. Gradient FFT-NCC (Scharr magnitude)
  3. Orientation Energy FFT-NCC (Structure tensor coherence)
  4. High-Pass Structural Map FFT-NCC (Illumination/charging suppression)
  5. 4 Local Sub-Template Structural Anchor Views
Computes Candidate UNION, Multi-View Voting, and Spatial NMS to extract Top-30 spatial candidates.
"""

import cv2
import numpy as np

from generate_pace_dataset import normalize_intensity
from inference_car import extract_gradient
from orientation_features import compute_highpass_map, compute_orientation_energy
from anchor_retrieval_v7 import retrieve_anchor_candidates


def extract_spatial_peaks(corr_plane: np.ndarray, k_max: int = 10, offset: float = 50.0, source_label: str = "map") -> list[dict]:
    work = corr_plane.copy()
    ch, cw = work.shape
    peaks = []

    for _ in range(k_max):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + offset
        cy = py + offset
        peaks.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "score": float(max_val),
            "source": source_label,
            "corr_plane": corr_plane,
        })
        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0

    return peaks


def redundant_multi_view_retrieval(search_img: np.ndarray, ref_img: np.ndarray,
                                    use_anchors: bool = True,
                                    k_top: int = 30) -> tuple[list[dict], dict]:
    """
    Computes redundant multi-view candidate retrieval:
    Spatial Union across Intensity, Gradient, Orientation, High-Pass, and Local Anchors.
    """
    sh, sw = search_img.shape

    # Physical Normalization (1000x1000 ref downsampled to 100x100 template)
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)
    ref_100_proc = cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA)

    # Representation 1: Intensity
    search_norm = normalize_intensity(search_proc)
    ref_100_norm = normalize_intensity(ref_100_proc)
    c_i = cv2.matchTemplate(search_norm, ref_100_norm, cv2.TM_CCOEFF_NORMED)
    peaks_i = extract_spatial_peaks(c_i, k_max=10, source_label="intensity")

    # Representation 2: Gradient
    search_grad = extract_gradient(search_proc)
    ref_100_grad = extract_gradient(ref_100_proc)
    c_g = cv2.matchTemplate(search_grad, ref_100_grad, cv2.TM_CCOEFF_NORMED)
    peaks_g = extract_spatial_peaks(c_g, k_max=10, source_label="gradient")

    # Representation 3: Orientation Energy
    search_ori = compute_orientation_energy(search_proc)
    ref_100_ori = compute_orientation_energy(ref_100_proc)
    c_o = cv2.matchTemplate(search_ori, ref_100_ori, cv2.TM_CCOEFF_NORMED)
    peaks_o = extract_spatial_peaks(c_o, k_max=10, source_label="orientation")

    # Representation 4: High-Pass Structural Map
    search_hp = compute_highpass_map(search_proc)
    ref_100_hp = compute_highpass_map(ref_100_proc)
    c_h = cv2.matchTemplate(search_hp, ref_100_hp, cv2.TM_CCOEFF_NORMED)
    peaks_h = extract_spatial_peaks(c_h, k_max=10, source_label="highpass")

    all_peaks = peaks_i + peaks_g + peaks_o + peaks_h

    # Add Local Sub-Template Structural Anchor Views
    if use_anchors:
        anchor_peaks = retrieve_anchor_candidates(search_img, ref_img, k_max_per_anchor=5)
        all_peaks += anchor_peaks

    # Candidate UNION + Multi-View Voting & Spatial NMS
    union_candidates = []
    nms_radius = 12.0

    for p in all_peaks:
        # Check if already present in union
        match_c = None
        for u in union_candidates:
            if np.hypot(p["cx"] - u["cx"], p["cy"] - u["cy"]) < nms_radius:
                match_c = u
                break

        if match_c is None:
            new_c = {
                "cx": p["cx"],
                "cy": p["cy"],
                "peak_x": int(round(p["cx"] - 50.0)),
                "peak_y": int(round(p["cy"] - 50.0)),
                "sources": [p["source"]],
                "score_i": float(c_i[int(round(p["cy"] - 50.0)), int(round(p["cx"] - 50.0))]) if 0 <= int(round(p["cy"] - 50.0)) < c_i.shape[0] and 0 <= int(round(p["cx"] - 50.0)) < c_i.shape[1] else p["score"],
                "max_feat_score": p["score"],
                "vote_count": 1,
                "corr_plane": c_i,
            }
            union_candidates.append(new_c)
        else:
            match_c["sources"].append(p["source"])
            match_c["vote_count"] += 1
            match_c["max_feat_score"] = max(match_c["max_feat_score"], p["score"])

    # Multi-View Voting Score = vote_count + max_feat_score
    for c in union_candidates:
        c["ncc_score"] = c["score_i"]
        c["v7_rank_score"] = c["vote_count"] + 0.5 * c["max_feat_score"] + 0.5 * c["score_i"]

    union_candidates.sort(key=lambda c: c["v7_rank_score"], reverse=True)
    top_candidates = union_candidates[:k_top]

    debug_meta = {
        "num_total_peaks": len(all_peaks),
        "num_union_candidates": len(union_candidates),
        "num_top_candidates": len(top_candidates),
    }

    return top_candidates, debug_meta
