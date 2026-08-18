"""
Drift-Sense V5: Multi-Anchor Geometric Consensus & Structural Verification Pipeline
Features:
1. Fast Scale & Rotation Coarse-to-Fine Estimation
2. Multi-Anchor Distinctive Patch Selection with Self-Similarity Repetition Penalty
3. Multi-Anchor + Whole-Template Geometric Consensus Retrieval
4. Multi-Feature Structural Verification (Intensity ZNCC + Gradient Coherence + Orientation + Frequency Pitch + Local Phase)
5. Calibrated Periodic Ambiguity Detection (Peak Spread Delta-S < tau -> Center Proximity Prior)
6. 2D Paraboloid Subpixel Refinement
"""

import os
import sys
import argparse
import json
import time
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


def extract_gradient_map(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts Scharr gradient magnitude [0, 1] and orientation angle [-pi, pi]."""
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    g_ang = np.arctan2(gy, gx)
    max_val = g_mag.max()
    if max_val > 1e-6:
        g_mag /= max_val
    return g_mag.astype(np.float32), g_ang.astype(np.float32)


def compute_frequency_pitch_spectrum(patch: np.ndarray) -> np.ndarray:
    """Computes 1D radial and directional frequency energy profile of a patch."""
    h, w = patch.shape
    if h < 16 or w < 16:
        return np.zeros(16, dtype=np.float32)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    patch_w = (patch.astype(np.float32) - np.mean(patch)) * win
    f = np.fft.fftshift(np.fft.fft2(patch_w))
    psd = np.abs(f) ** 2
    # 1D column and row projections of power spectrum (pitch peaks)
    proj_x = np.sum(psd, axis=0)
    proj_y = np.sum(psd, axis=1)
    feat = np.concatenate([proj_x[:8], proj_y[:8]])
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat /= norm
    return feat.astype(np.float32)


def extract_distinct_spatial_peaks(corr_map: np.ndarray, top_k: int = 6, min_dist: int = 15) -> list[dict]:
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


def select_distinctive_anchors(
    ref_100: np.ndarray,
    patch_size: int = 36,
    stride: int = 12,
    num_anchors: int = 3
) -> list[dict]:
    """
    Selects top non-overlapping anchor patches with high information and low self-similarity.
    """
    ref_norm = normalize_intensity(ref_100)
    ref_grad, _ = extract_gradient_map(ref_100)
    h, w = ref_norm.shape
    ref_cx, ref_cy = w / 2.0, h / 2.0

    candidates = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p_int = ref_norm[y:y+patch_size, x:x+patch_size]
            p_grad = ref_grad[y:y+patch_size, x:x+patch_size]

            grad_energy = float(np.mean(p_grad) + 0.5 * np.std(p_grad))
            int_std = float(np.std(p_int))
            info_score = grad_energy + int_std

            if info_score < 0.05:
                continue

            res_self = cv2.matchTemplate(ref_norm, p_int, cv2.TM_CCOEFF_NORMED)
            rh, rw = res_self.shape
            mask = np.ones((rh, rw), dtype=bool)
            y_min, y_max = max(0, y - 6), min(rh, y + 7)
            x_min, x_max = max(0, x - 6), min(rw, x + 7)
            mask[y_min:y_max, x_min:x_max] = False

            off_center = res_self[mask]
            max_self_sim = float(np.max(off_center)) if len(off_center) > 0 else 0.0
            uniqueness = max(0.01, 1.0 - max_self_sim)
            distinctiveness = (info_score * (uniqueness ** 1.5)) / (max_self_sim + 0.15)

            offset_x = (x + patch_size / 2.0) - ref_cx
            offset_y = (y + patch_size / 2.0) - ref_cy

            candidates.append({
                "x": x, "y": y, "patch_size": patch_size,
                "offset_x": offset_x, "offset_y": offset_y,
                "patch_int": p_int, "patch_grad": p_grad,
                "distinctiveness": distinctiveness,
                "max_self_sim": max_self_sim
            })

    candidates.sort(key=lambda c: c["distinctiveness"], reverse=True)
    selected_anchors = []
    for cand in candidates:
        cx, cy = cand["x"] + patch_size / 2.0, cand["y"] + patch_size / 2.0
        if not any(np.hypot(cx - (a["x"] + patch_size / 2.0), cy - (a["y"] + patch_size / 2.0)) < (patch_size * 0.55) for a in selected_anchors):
            selected_anchors.append(cand)
        if len(selected_anchors) >= num_anchors:
            break

    if len(selected_anchors) == 0:
        for qx, qy in [(15, 15), (50, 15), (15, 50), (50, 50)]:
            p_int = ref_norm[qy:qy+patch_size, qx:qx+patch_size]
            p_grad = ref_grad[qy:qy+patch_size, qx:qx+patch_size]
            selected_anchors.append({
                "x": qx, "y": qy, "patch_size": patch_size,
                "offset_x": (qx + patch_size / 2.0) - ref_cx,
                "offset_y": (qy + patch_size / 2.0) - ref_cy,
                "patch_int": p_int, "patch_grad": p_grad,
                "distinctiveness": 1.0, "max_self_sim": 0.5
            })

    return selected_anchors


def local_phase_correlation(ref_patch: np.ndarray, search_patch: np.ndarray) -> tuple[float, float, float]:
    """Computes local phase correlation between reference and candidate search patch."""
    h, w = ref_patch.shape
    sh, sw = search_patch.shape
    if (h, w) != (sh, sw):
        ref_patch = cv2.resize(ref_patch, (sw, sh), interpolation=cv2.INTER_AREA)

    ref_f = ref_patch.astype(np.float32)
    search_f = search_patch.astype(np.float32)

    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    ref_win = (ref_f - np.mean(ref_f)) * hann
    search_win = (search_f - np.mean(search_f)) * hann

    G_a = np.fft.fft2(ref_win)
    G_b = np.fft.fft2(search_win)
    cross_power = G_a * np.conj(G_b)
    magnitude = np.abs(cross_power) + 1e-7
    r = np.real(np.fft.ifft2(cross_power / magnitude))
    r_shift = np.fft.fftshift(r)

    cy, cx = sh // 2, sw // 2
    win_r = 5
    sub_r = r_shift[max(0, cy - win_r):min(sh, cy + win_r + 1),
                    max(0, cx - win_r):min(sw, cx + win_r + 1)]

    _, max_val, _, max_loc = cv2.minMaxLoc(sub_r)
    dx = float(max_loc[0] - win_r)
    dy = float(max_loc[1] - win_r)
    phase_score = float(max_val)

    return dx, dy, phase_score


def subpixel_refine_2d(corr_plane: np.ndarray, int_x: int, int_y: int) -> tuple[float, float]:
    """Fits 2D quadratic paraboloid surface on 5x5 neighborhood around correlation peak."""
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)

    patch = corr_plane[int_y-2:int_y+3, int_x-2:int_x+3].astype(np.float64)
    y_coords, x_coords = np.mgrid[-2:3, -2:3]

    X_mat = np.column_stack([
        x_coords.ravel()**2,
        y_coords.ravel()**2,
        x_coords.ravel() * y_coords.ravel(),
        x_coords.ravel(),
        y_coords.ravel(),
        np.ones(25)
    ])
    Z_vec = patch.ravel()

    try:
        coeff, _, _, _ = np.linalg.lstsq(X_mat, Z_vec, rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c**2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = (c * e - 2 * b * d) / denom
            dy = (c * d - 2 * a * e) / denom
            dx = np.clip(dx, -1.5, 1.5)
            dy = np.clip(dy, -1.5, 1.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass

    return float(int_x), float(int_y)


def perform_drift_sense_v5(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    scales: tuple[float, ...] = (0.97, 1.00, 1.03),
    rotations: tuple[float, ...] = (-2.0, 0.0, 2.0),
    top_k_candidates: int = 10,
    ambiguity_tau: float = 0.035,
    use_multi_anchor: bool = True,
    verbose: bool = False
) -> tuple[float, float, dict]:
    """
    Main Drift-Sense V5 Inference Pipeline:
    1. Downsample reference to 100x100 and extract distinctive anchors.
    2. Multi-scale & micro-rotation retrieval with geometric consensus voting.
    3. Multi-feature structural verification (ZNCC + Gradient + Orientation + Pitch Spectrum + Local Phase).
    4. Periodic ambiguity detection (delta-S < tau -> apply Problem Statement center rule).
    5. Subpixel 2D paraboloid refinement.
    """
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0

    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)

    search_norm = normalize_intensity(search_proc)
    search_grad, search_ang = extract_gradient_map(search_proc)

    ref_100 = cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA)
    ref_100_norm = normalize_intensity(ref_100)
    ref_100_grad, ref_100_ang = extract_gradient_map(ref_100)
    ref_freq = compute_frequency_pitch_spectrum(ref_100_norm)

    anchors = select_distinctive_anchors(ref_100, patch_size=36, stride=12, num_anchors=3) if use_multi_anchor else []

    all_candidates = []

    # 1. Multi-Scale Whole-Template & Anchor Search
    for s in scales:
        tw = max(10, int(round(100 * s)))
        th = max(10, int(round(100 * s)))
        ref_s = cv2.resize(ref_proc, (tw, th), interpolation=cv2.INTER_AREA)

        for r in rotations:
            if abs(r) > 0.01:
                M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), r, 1.0)
                ref_r = cv2.warpAffine(ref_s, M, (tw, th), borderMode=cv2.BORDER_REFLECT)
            else:
                ref_r = ref_s

            r_norm = normalize_intensity(ref_r)
            r_grad, r_ang = extract_gradient_map(ref_r)

            c_i = cv2.matchTemplate(search_norm, r_norm, cv2.TM_CCOEFF_NORMED)
            c_g = cv2.matchTemplate(search_grad, r_grad, cv2.TM_CCOEFF_NORMED)
            c_combo = 0.55 * c_i + 0.45 * c_g

            peaks = extract_distinct_spatial_peaks(c_combo, top_k=4, min_dist=15)
            for p in peaks:
                all_candidates.append({
                    "x": p["x"] + tw / 2.0,
                    "y": p["y"] + th / 2.0,
                    "peak_x": p["x"],
                    "peak_y": p["y"],
                    "tw": tw,
                    "th": th,
                    "scale": s,
                    "rot": r,
                    "score": p["score"],
                    "corr_plane": c_combo,
                    "ref_n": r_norm,
                    "ref_g": r_grad,
                    "ref_ang": r_ang
                })

    # 2. Add Anchor Consensus Votes if enabled
    if use_multi_anchor and len(anchors) > 0:
        anchor_votes = []
        for a_idx, anchor in enumerate(anchors):
            p_int = anchor["patch_int"]
            p_grad = anchor["patch_grad"]
            off_x = anchor["offset_x"]
            off_y = anchor["offset_y"]
            a_weight = float(np.clip(anchor["distinctiveness"], 0.5, 2.5))

            for s in [0.97, 1.00, 1.03]:
                aw = max(8, int(round(anchor["patch_size"] * s)))
                ah = max(8, int(round(anchor["patch_size"] * s)))
                p_s_int = cv2.resize(p_int, (aw, ah), interpolation=cv2.INTER_AREA)
                p_s_grad = cv2.resize(p_grad, (aw, ah), interpolation=cv2.INTER_AREA)

                c_i = cv2.matchTemplate(search_norm, p_s_int, cv2.TM_CCOEFF_NORMED)
                c_g = cv2.matchTemplate(search_grad, p_s_grad, cv2.TM_CCOEFF_NORMED)
                c_combo = 0.55 * c_i + 0.45 * c_g

                peaks = extract_distinct_spatial_peaks(c_combo, top_k=3, min_dist=12)
                for p in peaks:
                    pred_x = (p["x"] + aw / 2.0) - off_x * s
                    pred_y = (p["y"] + ah / 2.0) - off_y * s
                    anchor_votes.append({
                        "anchor_id": a_idx,
                        "pred_x": pred_x,
                        "pred_y": pred_y,
                        "score": p["score"],
                        "weight": p["score"] * a_weight
                    })

        # Augment candidates with anchor agreement
        for cand in all_candidates:
            cx, cy = cand["x"], cand["y"]
            matched_anchors = set()
            vote_sum = 0.0
            for v in anchor_votes:
                if np.hypot(v["pred_x"] - cx, v["pred_y"] - cy) <= 8.0:
                    matched_anchors.add(v["anchor_id"])
                    vote_sum += v["weight"]
            cand["anchor_support"] = len(matched_anchors) / max(1, len(anchors))
            cand["score"] += 0.20 * cand["anchor_support"]

    # 3. Spatial Deduplication to Top-K Candidates
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    unique_candidates = []
    for cand in all_candidates:
        if not any(np.hypot(cand["x"] - u["x"], cand["y"] - u["y"]) < 12 for u in unique_candidates):
            unique_candidates.append(cand)
        if len(unique_candidates) >= top_k_candidates:
            break

    if not unique_candidates:
        return search_cx, search_cy, {"status": "NO_CANDIDATE"}

    # 4. Multi-Feature Structural Verification on Top-K Patches
    for c in unique_candidates:
        cx, cy = c["x"], c["y"]
        tw, th = c["tw"], c["th"]
        y1 = max(0, int(round(cy - th / 2.0)))
        y2 = min(sh, int(round(cy + th / 2.0)))
        x1 = max(0, int(round(cx - tw / 2.0)))
        x2 = min(sw, int(round(cx + tw / 2.0)))

        sp_n = search_norm[y1:y2, x1:x2]
        sp_g = search_grad[y1:y2, x1:x2]
        sp_ang = search_ang[y1:y2, x1:x2]

        if sp_n.shape != (th, tw):
            sp_n = cv2.resize(sp_n, (tw, th), interpolation=cv2.INTER_AREA)
            sp_g = cv2.resize(sp_g, (tw, th), interpolation=cv2.INTER_AREA)
            sp_ang = cv2.resize(sp_ang, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 4a. Local Phase correlation
        dx, dy, ps = local_phase_correlation(c["ref_n"], sp_n)

        # 4b. Gradient correlation
        g_corr = float(np.corrcoef(c["ref_g"].ravel(), sp_g.ravel())[0, 1])
        if np.isnan(g_corr):
            g_corr = 0.0

        # 4c. Orientation coherence
        ang_diff = np.abs(np.arctan2(np.sin(c["ref_ang"] - sp_ang), np.cos(c["ref_ang"] - sp_ang)))
        orient_score = float(1.0 - np.mean(ang_diff) / np.pi)

        # 4d. Frequency Pitch spectrum similarity
        cand_freq = compute_frequency_pitch_spectrum(sp_n)
        freq_sim = float(np.dot(ref_freq, cand_freq)) if np.linalg.norm(cand_freq) > 0 else 0.0

        # Composite Multi-Feature Score
        c["phase_score"] = ps
        c["grad_corr"] = g_corr
        c["orient_score"] = orient_score
        c["freq_sim"] = freq_sim
        c["dist_to_center"] = float(np.hypot(cx - search_cx, cy - search_cy))

        c["final_score"] = (
            0.45 * c["score"] +
            0.25 * g_corr +
            0.15 * max(0.0, ps) +
            0.10 * orient_score +
            0.05 * max(0.0, freq_sim)
        )

    # 5. Periodic Ambiguity & Center Disambiguation
    unique_candidates.sort(key=lambda c: c["final_score"], reverse=True)
    best = unique_candidates[0]
    is_ambiguous = False

    if len(unique_candidates) > 1:
        delta_s = unique_candidates[0]["final_score"] - unique_candidates[1]["final_score"]
        if delta_s < ambiguity_tau:
            is_ambiguous = True
            # Pool candidates within ambiguity margin of top1
            ambiguous_pool = [c for c in unique_candidates if (unique_candidates[0]["final_score"] - c["final_score"]) < ambiguity_tau]
            # Problem Statement rule: If more than one matching region is found, return the one closest to search center
            best = min(ambiguous_pool, key=lambda c: c["dist_to_center"])

    # 6. Subpixel Paraboloid Refinement
    sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
    final_x = float(sub_x + best["tw"] / 2.0)
    final_y = float(sub_y + best["th"] / 2.0)

    meta = {
        "x": round(final_x, 2),
        "y": round(final_y, 2),
        "scale": round(best["scale"], 4),
        "rotation": round(best["rot"], 2),
        "score": round(best["final_score"], 4),
        "raw_score": round(best["score"], 4),
        "grad_corr": round(best["grad_corr"], 4),
        "phase_score": round(best["phase_score"], 4),
        "orient_score": round(best["orient_score"], 4),
        "freq_sim": round(best["freq_sim"], 4),
        "is_ambiguous": is_ambiguous,
        "dist_to_center": round(best["dist_to_center"], 2),
        "status": "OK"
    }

    return final_x, final_y, meta


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense V5 Inference Engine")
    parser.add_argument("--reference", type=str, required=True, help="Path to 100x reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to 10x search image")
    parser.add_argument("--verbose", action="store_true", help="Output detailed JSON metadata")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    x, y, meta = perform_drift_sense_v5(ref_img, search_img, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(meta, indent=2))

    print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
