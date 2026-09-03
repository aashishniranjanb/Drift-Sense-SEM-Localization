import os
import sys
import cv2
import time
import subprocess
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.append('.')
sys.path.append('phase2')
sys.path.append('phase2/V39_1_SCALE')

WORKERS = max(1, os.cpu_count() - 1)

def init_worker():
    cv2.setNumThreads(1)

def process_single_pair(item: dict) -> dict:
    from phase2.V39_1_SCALE.v39_1_scale_refinement import evaluate_scale
    
    pid = item['pair_id']
    found = item['found']
    score = item['score']
    x_v39 = item['x']
    y_v39 = item['y']
    th_v39 = item['theta']
    sc_v39 = item['scale']
    
    if found == 0 or score <= 0.0:
        return {
            'pair_id': pid, 'x': 0.0, 'y': 0.0, 'theta': 0.0, 'scale': 0.0,
            'found': 0, 'score': 0.0, 'orig_scale': sc_v39, 'scale_score': 0.0, 'elapsed_ms': 0.0
        }
        
    t_start = time.time()
    ref_path = os.path.join('data/phase2_dev', item['reference_path'].replace('\\', '/'))
    srch_path = os.path.join('data/phase2_dev', item['search_path'].replace('\\', '/'))
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None or search_img is None:
        return {
            'pair_id': pid, 'x': x_v39, 'y': y_v39, 'theta': th_v39, 'scale': sc_v39,
            'found': found, 'score': score, 'orig_scale': sc_v39, 'scale_score': 0.0, 'elapsed_ms': 0.0
        }
        
    best_scale, best_score = evaluate_scale(
        ref_img, search_img, x_v39, y_v39, th_v39, sc_v39
    )
    t_el = (time.time() - t_start) * 1000.0
    
    return {
        'pair_id': pid,
        'x': x_v39,
        'y': y_v39,
        'theta': th_v39,
        'scale': best_scale,
        'found': found,
        'score': score,
        'orig_scale': sc_v39,
        'scale_score': best_score,
        'elapsed_ms': t_el
    }

def run_v39_1_benchmark():
    pairs_csv = 'data/phase2_dev/pairs.csv'
    v39_csv = 'phase2/V39_POSE/v39_predictions.csv'
    output_pred = 'phase2/V39_1_SCALE/v39_1_predictions.csv'
    output_res = 'phase2/V39_1_SCALE/v39_1_results.csv'
    
    df_pairs = pd.read_csv(pairs_csv)
    df_v39 = pd.read_csv(v39_csv)
    
    merged = pd.merge(df_pairs, df_v39, on='pair_id', suffixes=('', '_pred'))
    items = merged.to_dict('records')
    
    print(f"Running V39.1 Scale Refinement Benchmark on {len(items)} pairs with {WORKERS} workers...", flush=True)
    t0 = time.time()
    
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as executor:
        for i, res in enumerate(executor.map(process_single_pair, items), start=1):
            records.append(res)
            if i % 30 == 0 or i == len(items):
                el = time.time() - t0
                rate = i / el
                eta = (len(items) - i) / rate if rate > 0 else 0
                print(f"[{i}/{len(items)}] {rate:.2f} pairs/s | ETA: {eta:.1f}s", flush=True)
                
    total_time = time.time() - t0
    avg_runtime = total_time / len(items)
    print(f"\nV39.1 Execution Complete in {total_time:.2f}s (Avg {avg_runtime:.3f}s / pair)\n", flush=True)
    
    df_results = pd.DataFrame(records)
    df_results.to_csv(output_res, index=False)
    
    df_pred = df_results[['pair_id', 'x', 'y', 'theta', 'scale', 'found', 'score']]
    df_pred.to_csv(output_pred, index=False)
    
    python_exe = sys.executable
    bench_res = subprocess.run([
        python_exe,
        'phase2/benchmark_phase2.py',
        '--input-csv', pairs_csv,
        '--predictions-csv', output_pred
    ], capture_output=True, text=True)
    
    print("--- OFFICIAL BENCHMARK OUTPUT ---")
    print(bench_res.stdout)
    
    with open('phase2/V39_1_SCALE/V39_1_REPORT.md', 'w') as f:
        f.write(f"# V39.1 Scale Refinement\n\n`	ext\n{bench_res.stdout}\n`\n")

if __name__ == '__main__':
    cv2.setNumThreads(1)
    run_v39_1_benchmark()
