import os
import sys
import cv2
import numpy as np
import pandas as pd

# Path setup
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "phase2"))
sys.path.append(os.path.join(root_dir, "fallbacks"))
sys.path.append(os.path.join(root_dir, "production_engine"))
sys.path.append(os.path.join(root_dir, "team", "akhilesh-localization"))

from fallbacks.pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from family_clustering import cluster_replica_families
from spatial_fingerprint import compute_spatial_fingerprint
from context_matcher import verify_candidate_context
from inference_phase2 import (
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation
)

def extract_candidate_features(c, corr_plane, rotated_template, ref_img, search_img, est_scale, est_theta, all_candidates):
    sh, sw = search_img.shape[:2]
    search_cx, search_cy = sw / 2.0, sh / 2.0
    th, tw = rotated_template.shape[:2]
    
    px, py = c["peak_x"], c["peak_y"]
    cx, cy = c["cx"], c["cy"]
    
    y1, y2 = max(0, int(py)), min(sh, int(py + th))
    x1, x2 = max(0, int(px)), min(sw, int(px + tw))
    search_crop = search_img[y1:y2, x1:x2]
    
    psr, _, _ = compute_psr(corr_plane, px, py)
    ctx_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
    
    phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
    if search_crop.shape == (th, tw):
        phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_template, search_crop)
        
    ssd = 0.0
    if search_crop.shape == (th, tw):
        ssd = float(np.mean((search_crop.astype(np.float32) - rotated_template.astype(np.float32)) ** 2))
        
    phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
    dist_to_center = float(np.hypot(cx - search_cx, cy - search_cy))
    
    fam_members = [m for m in all_candidates if m.get("family_id") == c.get("family_id")]
    fp = compute_spatial_fingerprint(search_img, cx, cy, est_scale, fam_members)
    
    return {
        "cx": cx,
        "cy": cy,
        "corr_score": c.get("corr_score", 0.0),
        "rescue_score": c.get("rescue_score", 0.0),
        "psr": psr,
        "phase_residual": phase_residual,
        "phase_penalty": phase_penalty,
        "context_32": ctx_res.get("s32", 0.0),
        "context_64": ctx_res.get("s64", 0.0),
        "context_128": ctx_res.get("s128", 0.0),
        "context_combined": ctx_res.get("combined", 0.0),
        "ssd": ssd,
        "dist_to_center": dist_to_center,
        "nearest_edge_dist": fp.get("nearest_edge_dist", 0.0),
        "nearest_cut_dist": fp.get("nearest_cut_dist", 0.0),
        "row_spacing": fp.get("row_spacing", 0.0),
        "col_spacing": fp.get("col_spacing", 0.0),
        "local_density": fp.get("local_density", 0.0),
        "family_id": c.get("family_id", -1),
        "family_population": c.get("family_population", 1)
    }
