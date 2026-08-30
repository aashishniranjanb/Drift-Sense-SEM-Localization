import argparse
import pandas as pd
import sys
import os
import cv2

# Add production_engine and parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "production_engine"))

from production_runner import run_production_localization

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense++ V8 Phase 2 Registration Entry Point")
    parser.add_argument("--input", required=True, help="Path to input pairs.csv")
    parser.add_argument("--output", required=True, help="Path to output predictions.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    results = []
    
    for idx, row in df.iterrows():
        pair_id = row["pair_id"]
        ref_path = row["reference_path"]
        search_path = row["search_path"]
        
        # Resolve paths relative to the input CSV directory if they are relative
        csv_dir = os.path.dirname(os.path.abspath(args.input))
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(csv_dir, ref_path)
        if not os.path.isabs(search_path):
            search_path = os.path.join(csv_dir, search_path)
            
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            print(f"Error loading images for {pair_id}", file=sys.stderr)
            pred = {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0}
        else:
            # Perform localization and pose/presence recovery using Aashish production integration runner
            pred = run_production_localization(ref_img, search_img, verbose=False)
        
        results.append({
            "pair_id": pair_id,
            "x": pred["x"],
            "y": pred["y"],
            "theta": pred["theta"],
            "scale": pred["scale"],
            "found": pred["found"],
            "score": pred["score"]
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Registration complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
