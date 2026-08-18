import pandas as pd
import numpy as np
import cv2
from inference import extract_distinct_spatial_peaks

df = pd.read_csv('data/benchmark_120/manifest.csv')
print("Analyzing samples where True candidate was retrieved in Top-10 but not ranked #1:")
print("-" * 85)

count = 0
for idx, row in df.iterrows():
    ref = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
    
    ref_100 = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA)
    corr = cv2.matchTemplate(search.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    peaks = extract_distinct_spatial_peaks(corr, top_k=10, min_distance=15)
    
    dists = [np.hypot(p['x'] + 50 - gt_x, p['y'] + 50 - gt_y) for p in peaks]
    match_indices = [i for i, d in enumerate(dists) if d <= 5.0]
    if len(match_indices) > 0 and match_indices[0] > 0:
        count += 1
        true_rank = match_indices[0]
        top1_score = peaks[0]['score']
        true_score = peaks[true_rank]['score']
        diff = row['difficulty']
        arch = row['architecture']
        pred1_x = peaks[0]['x'] + 50
        pred1_y = peaks[0]['y'] + 50
        dx = pred1_x - gt_x
        dy = pred1_y - gt_y
        dist = np.hypot(dx, dy)
        print(f"Sample {idx:03d} [{diff:<11s} {arch:<12s}]: True Rank #{true_rank+1:<2d} | Top1={top1_score:.4f} True={true_score:.4f} (Delta={top1_score-true_score:.4f}) | Shift: dx={dx:+.1f}, dy={dy:+.1f}, dist={dist:.1f}px")

print("-" * 85)
print(f"Total cases where True match is in Top-10 but not #1: {count}")
