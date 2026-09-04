"""
TEST LATTICE PROBE GRID EXTENSION ON PAIR_031
==============================================
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from build_retrieval_v2 import extract_multi_source_union, estimate_local_pitch, subpixel_peak_refine

def main():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v54_pred = pd.read_csv("FINAL_SUBMISSION_GOLDEN/predictions.csv")

    row = pairs_df[pairs_df["pair_id"] == "pair_031"].iloc[0]
    v54_r = v54_pred[v54_pred["pair_id"] == "pair_031"].iloc[0]

    ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
    srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)

    scale_eval = float(v54_r["scale"]) if float(v54_r["scale"]) > 0.01 else 10.0
    theta_eval = float(v54_r["theta"])

    tw = int(round(ref.shape[1] / scale_eval))
    th = int(round(ref.shape[0] / scale_eval))
    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, theta_eval) if abs(theta_eval) > 0.01 else tpl
    corr_plane = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # Estimate local pitch around top peak (518.93, 475.33)
    anchor_px, anchor_py = 518.93 - tw/2.0, 475.33 - th/2.0
    pitch = estimate_local_pitch(corr_plane, anchor_px, anchor_py, search_radius=120)

    print("=" * 70)
    print("      LATTICE PITCH ESTIMATION FOR PAIR_031")
    print("=" * 70)
    if pitch:
        print(f"  Horizontal pitch vector (vx_x, vx_y): ({pitch['vx_x']:.2f}, {pitch['vx_y']:.2f})")
        print(f"  Vertical pitch vector (vy_x, vy_y): ({pitch['vy_x']:.2f}, {pitch['vy_y']:.2f})")

        # Test lattice step (0, +1)
        probe_px = anchor_px + pitch["vy_x"]
        probe_py = anchor_py + pitch["vy_y"]
        sp_x, sp_y = subpixel_peak_refine(corr_plane, probe_px, probe_py)
        probe_cx, probe_cy = sp_x + tw/2.0, sp_y + th/2.0
        err = np.hypot(probe_cx - gt_x, probe_cy - gt_y)
        corr_val = float(corr_plane[int(round(sp_y)), int(round(sp_x))]) if (0 <= int(round(sp_y)) < corr_plane.shape[0] and 0 <= int(round(sp_x)) < corr_plane.shape[1]) else 0.0

        print(f"\n  +1 Vertical Lattice Probe: ({probe_cx:.2f}, {probe_cy:.2f})")
        print(f"  GT Target: ({gt_x:.2f}, {gt_y:.2f})")
        print(f"  Error to GT: {err:.4f} px  (Correlation: {corr_val:.4f})")

if __name__ == "__main__":
    main()
