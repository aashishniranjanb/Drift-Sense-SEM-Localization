import argparse
import pandas as pd
import sys
import os
import cv2

# Add production_engine to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2", "V25_CHAMPIONSHIP"))

from v25_pipeline import run_v25_localization

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Root Execution")
    parser.add_argument("--input", required=True, help="Input CSV path (data/phase2_dev/pairs.csv)")
    parser.add_argument("--output", required=True, help="Output CSV path (predictions.csv)")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    results = []
    
    data_dir = os.path.dirname(os.path.abspath(args.input))
    if os.path.basename(data_dir) == "phase2_dev":
        data_dir = os.path.dirname(data_dir) # up to data/
        data_dir = os.path.join(data_dir, "phase2_dev")
        
    print(f"Running root registration on {len(df)} pairs...")
    
    for idx, row in df.iterrows():
        pair_id = row["pair_id"]
        set_type = row.get("set_type", "Unknown")
        
        ref_path = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])
        
        ref_img = cv2.imread(ref_path, 0)
        search_img = cv2.imread(search_path, 0)
        
        if ref_img is None or search_img is None:
            print(f"Failed to load images for {pair_id}")
            results.append({
                "pair_id": pair_id,
                "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0,
                "found": 0, "score": 0.0
            })
            continue
            
        # Run V25 localization
        pred = run_v25_localization(ref_img, search_img, verbose=False)
        
        results.append({
            "pair_id": pair_id,
            "x": pred["x"],
            "y": pred["y"],
            "theta": pred["theta"],
            "scale": pred["scale"],
            "found": pred["found"],
            "score": pred["score"]
        })
        print(f"Processed {idx+1}/{len(df)}: {pair_id} | score: {pred['score']:.2f}")
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Registration complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
