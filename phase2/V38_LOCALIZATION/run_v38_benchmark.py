import os
import sys
import cv2
import numpy as np
import pandas as pd
import time
import subprocess

sys.path.append('.')
sys.path.append('phase2')
sys.path.append('phase2/V38_LOCALIZATION')

from v38_pipeline import run_v38_localization

def run_v38_benchmark(pairs_csv='data/phase2_dev/pairs.csv', output_predictions='phase2/V38_LOCALIZATION/v38_predictions.csv'):
    df_pairs = pd.read_csv(pairs_csv)
    print(f"Running V38 Pipeline Evaluation on {len(df_pairs)} pairs...", flush=True)
    
    records = []
    t0 = time.time()
    for idx, row in df_pairs.iterrows():
        pid = row['pair_id']
        ref_path = os.path.join('data/phase2_dev', row['reference_path'].replace('\\', '/'))
        srch_path = os.path.join('data/phase2_dev', row['search_path'].replace('\\', '/'))
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        
        res = run_v38_localization(ref_img, search_img)
        res['pair_id'] = pid
        records.append(res)
        
        if (len(records)) % 20 == 0:
            print(f"[{len(records)}/{len(df_pairs)}] Processed {pid}...", flush=True)
            
    total_time = time.time() - t0
    avg_runtime = total_time / len(df_pairs)
    print(f"Inference complete in {total_time:.2f}s (Avg {avg_runtime:.3f}s / pair).", flush=True)
    
    df_pred = pd.DataFrame(records)[['pair_id', 'x', 'y', 'theta', 'scale', 'found', 'score']]
    df_pred.to_csv(output_predictions, index=False)
    
    # Run official benchmark evaluator
    res_bench = subprocess.run([
        'C:\\Users\\jacks\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe',
        'phase2/benchmark_phase2.py',
        '--input-csv', pairs_csv,
        '--predictions-csv', output_predictions
    ], capture_output=True, text=True)
    
    print("\n--- BENCHMARK OUTPUT ---")
    print(res_bench.stdout)
    
    # Generate V38_REPORT.md comparing V28-C and V38
    with open('phase2/V38_LOCALIZATION/V38_REPORT.md', 'w') as f:
        f.write("# V38 Local Pose Optimization Report\n\n")
        f.write(f"## Runtime Summary\n- **Total Pairs**: {len(df_pairs)}\n- **Total Time**: {total_time:.2f}s\n- **Avg Time/Pair**: {avg_runtime:.3f}s\n\n")
        f.write("## Benchmark Output\n```\n")
        f.write(res_bench.stdout)
        f.write("\n```\n")
    print("Report saved to phase2/V38_LOCALIZATION/V38_REPORT.md")

if __name__ == '__main__':
    run_v38_benchmark()
