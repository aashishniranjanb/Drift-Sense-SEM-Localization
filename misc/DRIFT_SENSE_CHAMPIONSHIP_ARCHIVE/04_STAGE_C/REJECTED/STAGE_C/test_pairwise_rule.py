import os
import json
import numpy as np
import pandas as pd

def main():
    cache_path = "FINAL_SUBMISSION/validation/championship/precomputed_cache.json"
    if not os.path.exists(cache_path):
        print("Waiting for cache...")
        return
        
    with open(cache_path, "r") as f:
        precomputed = json.load(f)
        
    for p in precomputed:
        pid = p["pair_id"]
        gt_found = p["gt_found"]
        gt_x, gt_y = p["gt_x"], p["gt_y"]
        v54_x, v54_y = p["v54_x"], p["v54_y"]
        v54_found, v54_score = p["v54_found"], p["v54_score"]
        v54_struct = p.get("v54_struct", -1.0)
        best_c = p["best_cand"]
        
        base_err = float(np.hypot(v54_x - gt_x, v54_y - gt_y)) if (gt_found == 1 and v54_x > 0.1) else -1.0
        
        if base_err > 5.0 and best_c is not None:
            best_st = best_c["struct_score"]
            best_err = float(np.hypot(best_c["cx"] - gt_x, best_c["cy"] - gt_y))
            print(f"{pid}: base_err={base_err:.1f} best_err={best_err:.1f} v54_st={v54_struct:.3f} best_st={best_st:.3f} delta={best_st - v54_struct:.3f}")
            
if __name__ == "__main__":
    main()
