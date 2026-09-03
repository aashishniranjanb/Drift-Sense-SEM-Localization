import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import subprocess
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Load benchmark scoring function
import sys
sys.path.append('phase2')
from benchmark_phase2 import evaluate_phase2

# Target benchmark scoring
def run_benchmark_on_df(df_cand, name):
    tmp_path = f'phase2/V28_CHAMPIONSHIP/{name}.csv'
    df_cand.to_csv(tmp_path, index=False)
    
    # Run evaluation script and capture stdout
    res = subprocess.run(['python', 'phase2/benchmark_phase2.py', '--input-csv', 'data/phase2_dev/pairs.csv', '--predictions-csv', tmp_path], capture_output=True, text=True)
    
    # Parse metrics
    loc_score = 0.0
    rej_f1 = 0.0
    spearman_rho = 0.0
    for line in res.stdout.split('\n'):
        if 'OFFICIAL WEIGHTED LOC SCORE' in line:
            import re
            m = re.search(r'([\d\.]+)%', line)
            if m: loc_score = float(m.group(1)) * 0.40
        elif 'Set C Rejection F1 Score:' in line:
            import re
            m = re.search(r'([\d\.]+)', line.split(':')[-1])
            if m: rej_f1 = float(m.group(1))
        elif 'Spearman Rank Correlation (rho):' in line:
            import re
            m = re.search(r'([\d\.]+)', line.split(':')[-1])
            if m: spearman_rho = float(m.group(1))
            
    rej_pts = rej_f1 * 15.0
    cal_pts = spearman_rho * 10.0
    pose_pts = 18.0
    eff_pts = 5.0
    doc_pts = 10.0
    total = loc_score + pose_pts + rej_pts + cal_pts + eff_pts + doc_pts
    
    print(f'{name:20s} | Loc: {loc_score:5.2f} | Pose: {pose_pts:5.2f} | Rej: {rej_pts:5.2f} (F1: {rej_f1:.4f}) | Cal: {cal_pts:5.2f} (rho: {spearman_rho:.4f}) | Total: {total:6.2f}')
    return total, loc_score, rej_pts, cal_pts

# Base V25
run_benchmark_on_df(p_thresh, 'V25_baseline')

# Strategy 1: Scale scores of rejected pairs so that accepted pairs (high confidence) stay high,
# but rejected pairs are mapped to a range below accepted pairs
for rej_val in [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8]:
    df_mod = p_thresh.copy()
    # Map rejected pairs to rej_val
    df_mod.loc[df_mod['found'] == 0, 'score'] = rej_val
    run_benchmark_on_df(df_mod, f'rej_fixed_{rej_val}')

# Strategy 2: Monotonic compression: accepted pairs mapped to [0.8, 1.0], rejected pairs mapped to [0.0, 0.79]
# preserving relative rank among accepted, and relative rank among rejected!
df_ranked = p_thresh.copy()
acc_mask = (df_ranked['found'] == 1)
rej_mask = (df_ranked['found'] == 0)

# Rank among accepted
df_ranked.loc[acc_mask, 'score'] = 0.80 + 0.20 * (p_raw.loc[acc_mask, 'score'] - p_raw.loc[acc_mask, 'score'].min()) / (p_raw.loc[acc_mask, 'score'].max() - p_raw.loc[acc_mask, 'score'].min())
# Rank among rejected
df_ranked.loc[rej_mask, 'score'] = 0.10 + 0.65 * (p_raw.loc[rej_mask, 'score'] - p_raw.loc[rej_mask, 'score'].min()) / (p_raw.loc[rej_mask, 'score'].max() - p_raw.loc[rej_mask, 'score'].min())
run_benchmark_on_df(df_ranked, 'strat2_ranked_two_tier')
