import argparse
import pandas as pd
import sys
import os
import cv2
import numpy as np

# Add project root and subdirectories to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "phase2"))
sys.path.append(os.path.join(ROOT_DIR, "phase2", "V25_CHAMPIONSHIP"))
sys.path.append(os.path.join(ROOT_DIR, "fallbacks"))

from V25_CHAMPIONSHIP.v25_pipeline import run_v25_localization
from pose_fallback import perform_pose_fallback_search
from pose_refinement import refine_pose

def get_grad(img):
    img_f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    mx = g.max()
    if mx > 1e-6: g /= mx
    return (g * 255).astype(np.uint8)

def run_rgb_localization(ref_color, search_color):
    ref_y = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    search_y = cv2.cvtColor(search_color, cv2.COLOR_BGR2GRAY)
    
    pose_i = perform_pose_fallback_search(ref_y, search_y)
    
    search_g = get_grad(search_y)
    template_g = get_grad(pose_i['best_template'])
    
    corr_plane_g = cv2.matchTemplate(search_g.astype(np.float32), template_g.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    
    corr_union = np.maximum(pose_i['corr_plane'], corr_plane_g)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr_union)
    
    rx, ry, _, _ = refine_pose(ref_y, search_y, pose_i['best_scale'], pose_i['best_theta'], max_loc[0], max_loc[1], corr_union)
    
    found = 1 if max_val > 0.4 else 0
    if found == 0:
        rx, ry, theta, scale = 0.0, 0.0, 0.0, 0.0
    else:
        theta = pose_i['best_theta']
        scale = pose_i['best_scale']
        
    return {
        "x": float(rx),
        "y": float(ry),
        "theta": float(theta),
        "scale": float(scale),
        "found": int(found),
        "score": float(max_val)
    }

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Execution")
    parser.add_argument("--input", required=True, help="Input CSV path (data/phase2_dev/pairs.csv)")
    parser.add_argument("--output", required=True, help="Output CSV path (predictions.csv)")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    results = []
    
    data_dir = os.path.dirname(os.path.abspath(args.input))
    if os.path.basename(data_dir) == "phase2_dev":
        data_dir = os.path.dirname(data_dir)
        data_dir = os.path.join(data_dir, "phase2_dev")
        
    print(f"Running registration on {len(df)} pairs...")
    
    for idx, row in df.iterrows():
        pair_id = row["pair_id"]
        
        ref_path = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])
        
        ref_color = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        search_color = cv2.imread(search_path, cv2.IMREAD_COLOR)
        
        if ref_color is None or search_color is None:
            results.append({
                "pair_id": pair_id,
                "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": 0.0
            })
            continue
            
        is_rgb = False
        if len(ref_color.shape) == 3 and ref_color.shape[2] == 3:
            b, g, r = cv2.split(ref_color)
            if not (np.array_equal(b, g) and np.array_equal(g, r)):
                is_rgb = True
                
        if is_rgb:
            pred = run_rgb_localization(ref_color, search_color)
        else:
            ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
            search_gray = cv2.cvtColor(search_color, cv2.COLOR_BGR2GRAY)
            pred = run_v25_localization(ref_gray, search_gray, verbose=False)
            
            # V28-C Rules
            if pred["score"] <= 0.873:
                pred["found"] = 0
                pred["x"] = 0.0
                pred["y"] = 0.0
                pred["theta"] = 0.0
                pred["scale"] = 0.0
                pred["score"] = 0.0
            else:
                pred["found"] = 1
                
            # V47 Validator
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), "V47_RESEARCH"))
                from v47_validator_prod import validate
                pred = validate(pred, ref_gray, search_gray)
            except Exception as e:
                pass

                
        results.append({
            "pair_id": pair_id,
            "x": pred["x"],
            "y": pred["y"],
            "theta": pred["theta"],
            "scale": pred["scale"],
            "found": pred["found"],
            "score": pred["score"]
        })
        print(f"Processed {idx+1}/{len(df)}: {pair_id} | score: {pred['score']:.2f} | RGB: {is_rgb}")
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Registration complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
