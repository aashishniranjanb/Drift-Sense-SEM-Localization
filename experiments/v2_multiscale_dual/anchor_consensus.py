"""
Multi-Anchor Consensus Retrieval Engine for Semiconductor SEM Localization
Identifies locally informative, low-self-similarity anchor regions in the reference template,
retrieves them independently across the search image, and aggregates their geometric consensus votes
to suppress periodic lattice false matches and maximize candidate retrieval recall.
"""

import numpy as np
import cv2


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalizes image intensity to float32 [0, 1] with percentile scaling."""
    img_f = image.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (1, 99))
    if p_high > p_low:
        img_norm = np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img_norm = img_f / 255.0
    return img_norm.astype(np.float32)


def extract_gradient_map(image: np.ndarray) -> np.ndarray:
    """Extracts Scharr gradient magnitude map normalized to [0, 1]."""
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    max_val = g_mag.max()
    if max_val > 1e-6:
        g_mag /= max_val
    return g_mag.astype(np.float32)


def select_distinctive_anchors(
    ref_100: np.ndarray,
    patch_size: int = 36,
    stride: int = 12,
    num_anchors: int = 4
) -> list[dict]:
    """
    Divides the 100x100 reference template into sliding patches and ranks them by distinctiveness:
    Distinctiveness = (Information_Entropy * Gradient_Energy) / (Max_Off_Center_Self_Similarity + epsilon)
    """
    ref_norm = normalize_intensity(ref_100)
    ref_grad = extract_gradient_map(ref_100)
    h, w = ref_norm.shape
    ref_cx, ref_cy = w / 2.0, h / 2.0

    candidates = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p_int = ref_norm[y:y+patch_size, x:x+patch_size]
            p_grad = ref_grad[y:y+patch_size, x:x+patch_size]

            # 1. Local information content
            grad_energy = float(np.mean(p_grad) + 0.5 * np.std(p_grad))
            int_std = float(np.std(p_int))
            info_score = grad_energy + int_std

            if info_score < 0.05:
                continue

            # 2. Self-similarity across reference template
            res_self = cv2.matchTemplate(ref_norm, p_int, cv2.TM_CCOEFF_NORMED)
            rh, rw = res_self.shape

            # Mask out the true patch neighborhood (+- 6 px)
            mask = np.ones((rh, rw), dtype=bool)
            y_min = max(0, y - 6)
            y_max = min(rh, y + 7)
            x_min = max(0, x - 6)
            x_max = min(rw, x + 7)
            mask[y_min:y_max, x_min:x_max] = False

            off_center_scores = res_self[mask]
            max_self_sim = float(np.max(off_center_scores)) if len(off_center_scores) > 0 else 0.0
            max_self_sim = max(0.0, max_self_sim)

            # Saliency formula: Reward high information & structural uniqueness, heavily penalize periodic repetition
            uniqueness = max(0.01, 1.0 - max_self_sim)
            distinctiveness = (info_score * (uniqueness ** 1.5)) / (max_self_sim + 0.15)

            # Center offset relative to 100x100 template center (50, 50)
            offset_x = (x + patch_size / 2.0) - ref_cx
            offset_y = (y + patch_size / 2.0) - ref_cy

            candidates.append({
                "x": x,
                "y": y,
                "patch_size": patch_size,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "patch_int": p_int,
                "patch_grad": p_grad,
                "distinctiveness": distinctiveness,
                "max_self_sim": max_self_sim,
                "info_score": info_score
            })

    # Sort candidates by distinctiveness
    candidates.sort(key=lambda c: c["distinctiveness"], reverse=True)

    # Select top non-overlapping anchors
    selected_anchors = []
    for cand in candidates:
        cx = cand["x"] + patch_size / 2.0
        cy = cand["y"] + patch_size / 2.0
        if not any(np.hypot(cx - (a["x"] + patch_size / 2.0), cy - (a["y"] + patch_size / 2.0)) < (patch_size * 0.55) for a in selected_anchors):
            selected_anchors.append(cand)
        if len(selected_anchors) >= num_anchors:
            break

    # If no distinctive anchors found (e.g. completely flat), fallback to quadrant centers
    if len(selected_anchors) == 0:
        for qx, qy in [(15, 15), (50, 15), (15, 50), (50, 50)]:
            p_int = ref_norm[qy:qy+patch_size, qx:qx+patch_size]
            p_grad = ref_grad[qy:qy+patch_size, qx:qx+patch_size]
            selected_anchors.append({
                "x": qx, "y": qy, "patch_size": patch_size,
                "offset_x": (qx + patch_size / 2.0) - ref_cx,
                "offset_y": (qy + patch_size / 2.0) - ref_cy,
                "patch_int": p_int, "patch_grad": p_grad,
                "distinctiveness": 1.0, "max_self_sim": 0.5, "info_score": 1.0
            })

    return selected_anchors


def extract_peaks_nms(corr_map: np.ndarray, top_k: int = 8, min_dist: int = 15) -> list[dict]:
    """Extracts top distinct spatial peaks with non-maximum suppression."""
    work = corr_map.copy()
    h, w = work.shape
    peaks = []
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        peaks.append({"x": int(px), "y": int(py), "score": float(max_val)})
        y1, y2 = max(0, py - min_dist), min(h, py + min_dist + 1)
        x1, x2 = max(0, px - min_dist), min(w, px + min_dist + 1)
        work[y1:y2, x1:x2] = -999.0
    return peaks


def retrieve_multi_anchor_consensus(
    ref_100: np.ndarray,
    search_img: np.ndarray,
    scales: tuple[float, ...] = (0.96, 1.00, 1.04),
    top_k_candidates: int = 10,
    cluster_radius: float = 7.0
) -> tuple[list[dict], list[dict]]:
    """
    Executes Multi-Anchor Consensus Retrieval:
    1. Selects distinctive anchors in the reference template.
    2. Matches each anchor across the search image (Intensity + Gradient).
    3. Converts each anchor match to a predicted reference center: (x_match - offset_x, y_match - offset_y).
    4. Gathers geometric consensus clusters.
    5. Returns ranked consensus candidates and selected anchor metadata.
    """
    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    search_norm = normalize_intensity(search_proc)
    search_grad = extract_gradient_map(search_proc)

    anchors = select_distinctive_anchors(ref_100, patch_size=36, stride=12, num_anchors=4)
    ref_norm = normalize_intensity(ref_100)
    ref_grad = extract_gradient_map(ref_100)

    votes = []

    # 1. Anchor Votes
    for a_idx, anchor in enumerate(anchors):
        p_int = anchor["patch_int"]
        p_grad = anchor["patch_grad"]
        off_x = anchor["offset_x"]
        off_y = anchor["offset_y"]
        a_weight = float(np.clip(anchor["distinctiveness"], 0.5, 3.0))

        for s in scales:
            aw = max(8, int(round(anchor["patch_size"] * s)))
            ah = max(8, int(round(anchor["patch_size"] * s)))
            p_s_int = cv2.resize(p_int, (aw, ah), interpolation=cv2.INTER_AREA)
            p_s_grad = cv2.resize(p_grad, (aw, ah), interpolation=cv2.INTER_AREA)

            c_i = cv2.matchTemplate(search_norm, p_s_int, cv2.TM_CCOEFF_NORMED)
            c_g = cv2.matchTemplate(search_grad, p_s_grad, cv2.TM_CCOEFF_NORMED)
            c_combo = 0.55 * c_i + 0.45 * c_g

            peaks = extract_peaks_nms(c_combo, top_k=6, min_dist=12)
            for p in peaks:
                # Center of anchor match in search coords
                anchor_match_cx = p["x"] + aw / 2.0
                anchor_match_cy = p["y"] + ah / 2.0

                # Predicted Reference Center
                pred_ref_cx = anchor_match_cx - off_x * s
                pred_ref_cy = anchor_match_cy - off_y * s

                votes.append({
                    "anchor_id": a_idx,
                    "scale": s,
                    "pred_x": pred_ref_cx,
                    "pred_y": pred_ref_cy,
                    "score": p["score"],
                    "weight": p["score"] * a_weight
                })

    # 2. Whole-Template Global Matches
    whole_matches = []
    for s in scales:
        tw = max(10, int(round(100 * s)))
        th = max(10, int(round(100 * s)))
        r_s_int = cv2.resize(ref_norm, (tw, th), interpolation=cv2.INTER_AREA)
        r_s_grad = cv2.resize(ref_grad, (tw, th), interpolation=cv2.INTER_AREA)

        c_i = cv2.matchTemplate(search_norm, r_s_int, cv2.TM_CCOEFF_NORMED)
        c_g = cv2.matchTemplate(search_grad, r_s_grad, cv2.TM_CCOEFF_NORMED)
        c_combo = 0.55 * c_i + 0.45 * c_g

        peaks = extract_peaks_nms(c_combo, top_k=6, min_dist=15)
        for p in peaks:
            whole_matches.append({
                "x": p["x"] + tw / 2.0,
                "y": p["y"] + th / 2.0,
                "peak_x": p["x"],
                "peak_y": p["y"],
                "tw": tw,
                "th": th,
                "scale": s,
                "score": p["score"],
                "corr_plane": c_combo
            })

    # 3. Geometric Clustering of Votes
    # Seed candidate clusters from whole template peaks and dense anchor clusters
    clusters = []

    # Add all whole-template peaks as potential cluster centers
    for wm in whole_matches:
        clusters.append({
            "center_x": wm["x"],
            "center_y": wm["y"],
            "scale": wm["scale"],
            "tw": wm["tw"],
            "th": wm["th"],
            "peak_x": wm["peak_x"],
            "peak_y": wm["peak_y"],
            "whole_score": wm["score"],
            "corr_plane": wm["corr_plane"],
            "agreeing_anchors": set(),
            "anchor_vote_sum": 0.0,
            "anchor_scores": []
        })

    # Also seed clusters from top anchor votes not near existing clusters
    votes.sort(key=lambda v: v["weight"], reverse=True)
    for v in votes[:30]:
        if not any(np.hypot(v["pred_x"] - cl["center_x"], v["pred_y"] - cl["center_y"]) < cluster_radius for cl in clusters):
            s = v["scale"]
            tw = max(10, int(round(100 * s)))
            th = max(10, int(round(100 * s)))
            clusters.append({
                "center_x": v["pred_x"],
                "center_y": v["pred_y"],
                "scale": s,
                "tw": tw,
                "th": th,
                "peak_x": max(0, min(search_img.shape[1] - tw, int(round(v["pred_x"] - tw / 2.0)))),
                "peak_y": max(0, min(search_img.shape[0] - th, int(round(v["pred_y"] - th / 2.0)))),
                "whole_score": 0.0,
                "corr_plane": None,
                "agreeing_anchors": set(),
                "anchor_vote_sum": 0.0,
                "anchor_scores": []
            })

    # Associate anchor votes with clusters
    for cl in clusters:
        cx, cy = cl["center_x"], cl["center_y"]
        matched_anchors = set()
        matched_votes = []
        for v in votes:
            if np.hypot(v["pred_x"] - cx, v["pred_y"] - cy) <= cluster_radius:
                matched_anchors.add(v["anchor_id"])
                matched_votes.append(v)

        cl["agreeing_anchors"] = matched_anchors
        cl["num_anchors"] = len(matched_anchors)
        cl["anchor_vote_sum"] = sum(v["weight"] for v in matched_votes)

        # Composite Consensus Score:
        # High score if multiple independent anchors agree + strong whole template match
        num_agree = len(matched_anchors)
        cl["consensus_score"] = (
            0.40 * cl["whole_score"] +
            0.35 * (cl["anchor_vote_sum"] / max(1, len(anchors))) +
            0.25 * (num_agree / max(1, len(anchors)))
        )

    # Sort clusters by consensus score
    clusters.sort(key=lambda cl: cl["consensus_score"], reverse=True)

    # Deduplicate clusters
    unique_candidates = []
    for cl in clusters:
        if not any(np.hypot(cl["center_x"] - u["center_x"], cl["center_y"] - u["center_y"]) < 12.0 for u in unique_candidates):
            unique_candidates.append(cl)
        if len(unique_candidates) >= top_k_candidates:
            break

    return unique_candidates, anchors
