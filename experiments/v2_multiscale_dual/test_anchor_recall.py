"""
Test Multi-Anchor Consensus Candidate Retrieval Recall across 120 Benchmark Cases
Compares:
1. Baseline Whole-Template FFT-NCC Top-K Recall
2. Multi-Scale Dual Channel Top-K Recall
3. Multi-Anchor Consensus Top-K Recall
"""

import os
import time
import pandas as pd
import numpy as np
import cv2

from anchor_consensus import retrieve_multi_anchor_consensus, select_distinctive_anchors
from inference import extract_distinct_spatial_peaks, extract_gradient_map, normalize_intensity

MANIFEST_PATH = "data/benchmark_120/manifest.csv"
df = pd.read_csv(MANIFEST_PATH)

print(f"Evaluating Multi-Anchor Consensus Retrieval on {len(df)} samples...")

t_start = time.perf_counter()

recalls = {
    "Whole_FFT_NCC": {"Top-1": 0, "Top-3": 0, "Top-5": 0, "Top-10": 0},
    "Anchor_Consensus": {"Top-1": 0, "Top-3": 0, "Top-5": 0, "Top-10": 0}
}

diff_breakdown = {
    "Easy": {"Anchor_Top10": 0, "Total": 0},
    "Medium": {"Anchor_Top10": 0, "Total": 0},
    "Hard": {"Anchor_Top10": 0, "Total": 0},
    "Adversarial": {"Anchor_Top10": 0, "Total": 0},
}

for idx, row in df.iterrows():
    ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
    diff = row["difficulty"]
    diff_breakdown[diff]["Total"] += 1

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    # 1. Baseline Whole FFT-NCC
    corr_int = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    peaks_int = extract_distinct_spatial_peaks(corr_int, top_k=10, min_distance=15)
    cands_fft = [(p["x"] + 50.0, p["y"] + 50.0) for p in peaks_int]

    for k, name in [(1, "Top-1"), (3, "Top-3"), (5, "Top-5"), (10, "Top-10")]:
        if any(np.hypot(cx - gt_x, cy - gt_y) <= 5.0 for cx, cy in cands_fft[:k]):
            recalls["Whole_FFT_NCC"][name] += 1

    # 2. Multi-Anchor Consensus
    consensus_cands, _ = retrieve_multi_anchor_consensus(
        ref_100, search_img, scales=(0.96, 1.00, 1.04), top_k_candidates=10, cluster_radius=7.0
    )
    cands_anchor = [(c["center_x"], c["center_y"]) for c in consensus_cands]

    for k, name in [(1, "Top-1"), (3, "Top-3"), (5, "Top-5"), (10, "Top-10")]:
        if any(np.hypot(cx - gt_x, cy - gt_y) <= 5.0 for cx, cy in cands_anchor[:k]):
            recalls["Anchor_Consensus"][name] += 1
            if k == 10:
                diff_breakdown[diff]["Anchor_Top10"] += 1

t_elapsed = (time.perf_counter() - t_start) * 1000.0 / len(df)

total = len(df)
print("\n" + "=" * 80)
print("             RETRIEVAL RECALL COMPARISON (120 SAMPLES)             ")
print("=" * 80)

summary_rows = []
for m in ["Whole_FFT_NCC", "Anchor_Consensus"]:
    t1 = recalls[m]["Top-1"] / total * 100
    t3 = recalls[m]["Top-3"] / total * 100
    t5 = recalls[m]["Top-5"] / total * 100
    t10 = recalls[m]["Top-10"] / total * 100
    summary_rows.append({
        "Retrieval Method": m,
        "Top-1 (%)": round(t1, 2),
        "Top-3 (%)": round(t3, 2),
        "Top-5 (%)": round(t5, 2),
        "Top-10 (%)": round(t10, 2)
    })

df_res = pd.DataFrame(summary_rows)
print(df_res.to_string(index=False))
print("-" * 80)
print(f"Average Retrieval Latency per Sample: {t_elapsed:.2f} ms")

print("\nMulti-Anchor Top-10 Recall Breakdown by Difficulty:")
for d in ["Easy", "Medium", "Hard", "Adversarial"]:
    tot = diff_breakdown[d]["Total"]
    r10 = diff_breakdown[d]["Anchor_Top10"] / tot * 100 if tot > 0 else 0
    print(f"  [{d:<12s}]: Top-10 = {r10:.1f}% ({diff_breakdown[d]['Anchor_Top10']}/{tot})")
print("=" * 80)
