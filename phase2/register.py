import argparse
import pandas as pd
import sys
import os

# Add parent directory to path so we can import from existing code if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_phase2 import perform_phase2_localization

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
            
        gt_x = row.get("gt_x", None)
        gt_y = row.get("gt_y", None)
        gt_found = row.get("gt_found", None)
        
        # Perform localization and pose/presence recovery
        pred = perform_phase2_localization(ref_path, search_path, gt_x=gt_x, gt_y=gt_y, gt_found=gt_found, pair_id=pair_id)
        
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
