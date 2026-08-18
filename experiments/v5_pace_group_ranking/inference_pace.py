"""
Drift-Sense++ PACE: Process-Aware Contextual Embedding & Dual Subpixel Estimator Engine

Architecture:
1. Physical Normalization: 1000x1000 reference image downsampled to 100x100 template
2. Fast FFT-NCC Candidate Retrieval (Top-20 spatial peaks)
3. PACE Contextual Feature Extraction:
   - 64x64 Fine Local Structure
   - 128x128 Neighborhood Context
   - 4x32x32 Process-Variation Overlap Patches (Top, Bottom, Left, Right)
4. Group Candidate Ranking Scoring (PACE Neural Ranker)
5. Dual Subpixel Estimator Consensus:
   - Estimator A: Phase Correlation Peak
   - Estimator B: 2D Paraboloid Surface Fit Peak
   - Agreement Check: D = distance(Estimator A, Estimator B) < 2.0 px
6. Strict Conditional Periodicity Disambiguation
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
import torch

from pace_model import ProcessAwareContextEncoder
from generate_pace_dataset import extract_directional_overlaps, extract_patch_safe, normalize_intensity

DEFAULT_PACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pace_best.pt")


def extract_gradient_and_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    g_ang = np.arctan2(gy, gx)
    mx = g_mag.max()
    if mx > 1e-6:
        g_mag /= mx
    return g_mag.astype(np.float32), g_ang.astype(np.float32)


# ─── Dual Subpixel Estimator Engine ──────────────────────────────────────

def estimator_a_phase_correlation(ref_patch: np.ndarray, search_patch: np.ndarray) -> tuple[float, float, float]:
    """Estimator A: Analytic Phase Correlation Subpixel Estimation."""
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
    return float(max_loc[0] - win_r), float(max_loc[1] - win_r), float(max_val)


def estimator_b_paraboloid_fit(corr_plane: np.ndarray, int_x: int, int_y: int) -> tuple[float, float]:
    """Estimator B: 2D Paraboloid Surface Fitting with Clamped Offset [-0.5, 0.5] px."""
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)

    patch = corr_plane[int_y-2:int_y+3, int_x-2:int_x+3].astype(np.float64)
    y_coords, x_coords = np.mgrid[-2:3, -2:3]
    X_mat = np.column_stack([
        x_coords.ravel()**2, y_coords.ravel()**2,
        x_coords.ravel() * y_coords.ravel(),
        x_coords.ravel(), y_coords.ravel(), np.ones(25)
    ])
    Z_vec = patch.ravel()
    try:
        coeff, _, _, _ = np.linalg.lstsq(X_mat, Z_vec, rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c**2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = (c * e - 2 * b * d) / denom
            dy = (c * d - 2 * a * e) / denom
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass
    return float(int_x), float(int_y)


def evaluate_estimator_consensus(ref_100: np.ndarray, search_img: np.ndarray, candidate: dict) -> tuple[float, float, float, bool]:
    """
    Computes Dual Subpixel Estimator Agreement:
      D = distance(Estimator A Phase Peak, Estimator B Paraboloid Fit Peak)
    Returns (final_subpixel_x, final_subpixel_y, D, is_confident)
    """
    cx, cy = candidate["cx"], candidate["cy"]
    sh, sw = search_img.shape
    y1, y2 = max(0, int(round(cy - 50))), min(sh, int(round(cy + 50)))
    x1, x2 = max(0, int(round(cx - 50))), min(sw, int(round(cx + 50)))

    sp_n = search_img[y1:y2, x1:x2]
    if sp_n.shape != (100, 100):
        sp_n = cv2.resize(sp_n, (100, 100), interpolation=cv2.INTER_AREA)

    # Estimator A: Phase correlation
    dx_p, dy_p, p_score = estimator_a_phase_correlation(ref_100, sp_n)
    pos_a_x = cx + dx_p
    pos_a_y = cy + dy_p

    # Estimator B: 2D Paraboloid Fit
    pos_b_x, pos_b_y = estimator_b_paraboloid_fit(candidate["corr_plane"], candidate["peak_x"], candidate["peak_y"])
    pos_b_x += 50.0
    pos_b_y += 50.0

    # Estimator Distance
    D = float(np.hypot(pos_a_x - pos_b_x, pos_a_y - pos_b_y))
    is_confident = D <= 2.0

    # Weighted mean of Estimator A & Estimator B
    if is_confident:
        final_x = 0.5 * pos_a_x + 0.5 * pos_b_x
        final_y = 0.5 * pos_a_y + 0.5 * pos_b_y
    else:
        # Fallback to Estimator B (Paraboloid Fit)
        final_x, final_y = pos_b_x, pos_b_y

    return float(final_x), float(final_y), D, is_confident


# ─── PACE Neural Ranker Loading ──────────────────────────────────────────

_PACE_MODEL = None
_PACE_DEVICE = None


def load_pace_model(model_path: str = None) -> tuple:
    global _PACE_MODEL, _PACE_DEVICE
    if _PACE_MODEL is not None:
        return _PACE_MODEL, _PACE_DEVICE

    if model_path is None:
        model_path = DEFAULT_PACE_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        _PACE_DEVICE = device
        return None, device

    try:
        model = ProcessAwareContextEncoder()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        _PACE_MODEL = model
        _PACE_DEVICE = device
        return model, device
    except Exception:
        _PACE_DEVICE = device
        return None, device


# ─── Main PACE Inference Pipeline ────────────────────────────────────────

def perform_pace_localization(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    model_path: str = None,
    verbose: bool = False,
) -> tuple[float, float, dict]:
    t_start = time.perf_counter()
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0

    # ── Stage 0: Physical Normalization ──
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    # ── Stage 1: Fast Intensity ZNCC Correlation (Top-20 Candidates) ──
    c_i = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    work = c_i.copy()
    ch, cw = work.shape
    candidates = []
    for _ in range(20):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + 50.0
        cy = py + 50.0
        candidates.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "tw": 100, "th": 100,
            "scale": 1.0,
            "ncc_score": float(max_val),
            "corr_plane": c_i,
        })
        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0

    if len(candidates) == 0:
        return search_cx, search_cy, {"status": "NO_CANDIDATES", "latency_ms": 0}

    top1_ncc = candidates[0]["ncc_score"]
    top2_ncc = candidates[1]["ncc_score"] if len(candidates) > 1 else 0.0
    delta_s = top1_ncc - top2_ncc

    # ── FAST PATH: Unambiguous High-Confidence Match ──
    if delta_s >= 0.008 or top1_ncc >= 0.90:
        final_x, final_y, consensus_D, is_confident = evaluate_estimator_consensus(ref_100, search_img, candidates[0])
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return final_x, final_y, {
            "x": round(final_x, 2), "y": round(final_y, 2),
            "ncc_score": round(top1_ncc, 4),
            "delta_s": round(delta_s, 4),
            "consensus_D_px": round(consensus_D, 2),
            "is_confident": is_confident,
            "path": "FAST_PATH",
            "latency_ms": round(elapsed_ms, 2),
            "status": "OK"
        }

    # ── AMBIGUOUS PATH: PACE Neural Structural Re-Ranking ──
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
        for c in candidates:
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

        for i, c in enumerate(candidates):
            c["pace_score"] = float(scores[i])

        candidates.sort(key=lambda c: c["pace_score"], reverse=True)
    else:
        for c in candidates:
            c["pace_score"] = c["ncc_score"]

    top3 = candidates[:3]
    for c in top3:
        c["dist_to_center"] = float(np.hypot(c["cx"] - search_cx, c["cy"] - search_cy))

    # Periodicity Disambiguation (Center prior strictly on score ties)
    best = top3[0]
    is_ambiguous = False

    if len(top3) >= 2:
        final_delta = top3[0]["pace_score"] - top3[1]["pace_score"]
        cand_dist = np.hypot(top3[0]["cx"] - top3[1]["cx"], top3[0]["cy"] - top3[1]["cy"])

        if final_delta < 0.008 and 15.0 <= cand_dist <= 120.0:
            is_ambiguous = True
            ambiguity_pool = [c for c in top3 if (top3[0]["pace_score"] - c["pace_score"]) < 0.008]
            best = min(ambiguity_pool, key=lambda c: c["dist_to_center"])

    # Dual Subpixel Estimator Consensus for best candidate
    final_x, final_y, consensus_D, is_confident = evaluate_estimator_consensus(ref_100, search_img, best)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    meta = {
        "x": round(final_x, 2),
        "y": round(final_y, 2),
        "ncc_score": round(best["ncc_score"], 4),
        "pace_score": round(best.get("pace_score", 0.0), 4),
        "consensus_D_px": round(consensus_D, 2),
        "is_confident": is_confident,
        "is_ambiguous": is_ambiguous,
        "use_pace": use_pace,
        "dist_to_center": round(best.get("dist_to_center", 0.0), 2),
        "latency_ms": round(elapsed_ms, 2),
        "path": "AMBIGUOUS_PATH",
        "status": "OK",
    }

    return final_x, final_y, meta


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense++ PACE Production Inference")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    x, y, meta = perform_pace_localization(ref_img, search_img, model_path=args.model, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(meta, indent=2))

    print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
