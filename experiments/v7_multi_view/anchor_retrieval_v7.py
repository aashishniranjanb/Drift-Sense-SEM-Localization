"""
V7 Local Sub-Template Structural Anchor View Retrieval
Divides 1000x1000 reference into 4 overlapping 350x350 structural windows (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
and performs independent candidate center estimation.
"""

import cv2
import numpy as np


def extract_anchor_views(ref_img: np.ndarray, window_size: int = 350) -> list[dict]:
    """Extracts 4 overlapping structural anchor windows from 1000x1000 reference die."""
    h, w = ref_img.shape
    anchors = []

    # 4 Quadrant Anchors with overlap
    coords = [
        ("TL", 150, 150),
        ("TR", 150, 500),
        ("BL", 500, 150),
        ("BR", 500, 500),
    ]

    for label, y1, x1 in coords:
        patch = ref_img[y1:y1+window_size, x1:x1+window_size]
        # Resize to 35x35 for 10x search match
        patch_35 = cv2.resize(patch, (35, 35), interpolation=cv2.INTER_AREA)

        # Expected offset relative to 1000x1000 reference center
        anchor_cx_rel = (x1 + window_size / 2.0) - 500.0
        anchor_cy_rel = (y1 + window_size / 2.0) - 500.0

        anchors.append({
            "label": label,
            "patch_35": patch_35,
            "offset_x_search": anchor_cx_rel * 0.10,  # 10x scale conversion
            "offset_y_search": anchor_cy_rel * 0.10,
        })

    return anchors


def retrieve_anchor_candidates(search_img: np.ndarray, ref_img: np.ndarray, k_max_per_anchor: int = 5) -> list[dict]:
    """Performs local structural anchor template matching in search die and maps candidates back to full-die center."""
    anchors = extract_anchor_views(ref_img)
    search_f = search_img.astype(np.float32)

    anchor_cands = []

    for anc in anchors:
        patch_35 = anc["patch_35"].astype(np.float32)
        corr = cv2.matchTemplate(search_f, patch_35, cv2.TM_CCOEFF_NORMED)

        work = corr.copy()
        ch, cw = work.shape

        for _ in range(k_max_per_anchor):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val <= -1.0 or np.isnan(max_val):
                break
            px, py = max_loc
            # Anchor match center
            anc_cx = px + 17.5
            anc_cy = py + 17.5

            # Full-die center estimate = anchor_center - anchor_relative_offset
            die_cx = anc_cx - anc["offset_x_search"]
            die_cy = anc_cy - anc["offset_y_search"]

            anchor_cands.append({
                "cx": die_cx,
                "cy": die_cy,
                "score": float(max_val),
                "source": f"anchor_{anc['label']}",
            })

            y1, y2 = max(0, py - 8), min(ch, py + 9)
            x1, x2 = max(0, px - 8), min(cw, px + 9)
            work[y1:y2, x1:x2] = -999.0

    return anchor_cands
