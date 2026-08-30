import numpy as np
import cv2
import sys

# Import robust sequential pose search from phase2 baseline
sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search

def perform_pose_fallback_search(ref_img: np.ndarray, search_img: np.ndarray) -> dict:
    """
    Aashish Fallback Pose Estimator:
    Runs sequential coarse-to-fine scale and rotation search to recover (scale, theta).
    This decoupled approach is mathematically more robust against periodic clone trapping.
    """
    # 1. Coarse-to-fine scale search
    scale_res = coarse_to_fine_scale_search(ref_img, search_img)
    
    # 2. Coarse-to-fine rotation search
    rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
    
    return {
        "best_scale": float(scale_res["best_scale"]),
        "best_theta": float(rot_res["best_theta"]),
        "best_score": float(rot_res["best_score"]),
        "best_template": rot_res["rotated_template"],
        "corr_plane": rot_res["corr_plane"]
    }
