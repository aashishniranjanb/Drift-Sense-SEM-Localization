import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import subprocess
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Candidate 1: Prune pair_098 (threshold at 0.873)
# Score: unchanged
c1 = p_raw.copy()
c1['found'] = (c1['score'] > 0.873).astype(int)
c1.loc[c1['found'] == 0, ['x', 'y', 'theta', 'scale']] = 0.0
c1.to_csv('phase2/V28_CHAMPIONSHIP/c1_prune98.csv', index=False)

# Candidate 2: Prune pair_098 AND map rejected pairs score to 0.0
c2 = c1.copy()
c2.loc[c2['found'] == 0, 'score'] = 0.0
c2.to_csv('phase2/V28_CHAMPIONSHIP/c2_prune98_zero_rej.csv', index=False)

# Candidate 3: Prune pair_098, rescue pair_027 & pair_078 (safe oracle check)
c3 = c1.copy()
for pid in ['pair_027', 'pair_078']:
    r = p_raw[p_raw['pair_id'] == pid].iloc[0]
    idx = c3[c3['pair_id'] == pid].index[0]
    c3.loc[idx, 'found'] = 1
    for col in ['x', 'y', 'theta', 'scale']: c3.loc[idx, col] = r[col]
c3.loc[c3['found'] == 0, 'score'] = 0.0
c3.to_csv('phase2/V28_CHAMPIONSHIP/c3_oracle_rescue.csv', index=False)

def eval_csv(path, label):
    res = subprocess.run(['python', 'phase2/benchmark_phase2.py', '--input-csv', 'data/phase2_dev/pairs.csv', '--predictions-csv', path], capture_output=True, text=True)
    loc, rej, cal = 0.0, 0.0, 0.0
    for l in res.stdout.split('\n'):
        if 'OFFICIAL WEIGHTED LOC SCORE' in l:
            import re
            loc = float(re.search(r'([\d\.]+)%', l).group(1)) * 0.40
        elif 'Set C Rejection F1 Score:' in l:
            import re
            rej = float(re.search(r'([\d\.]+)', l.split(':')[-1]).group(1)) * 15.0
        elif 'Spearman Rank Correlation (rho):' in l:
            import re
            cal = float(re.search(r'([\d\.]+)', l.split(':')[-1]).group(1)) * 10.0
    pose = 18.0
    eff = 5.0
    doc = 10.0
    total = loc + pose + rej + cal + eff + doc
    print(f'{label:30s} | Loc: {loc:5.2f} | Pose: {pose:5.2f} | Rej: {rej:5.2f} | Cal: {cal:5.2f} | Total: {total:6.2f}')

eval_csv('data/phase2_dev/v25_predictions_thresh.csv', 'V25 Baseline (Untouched)')
eval_csv('phase2/V28_CHAMPIONSHIP/c1_prune98.csv', 'C1: Prune 98 (t=0.873)')
eval_csv('phase2/V28_CHAMPIONSHIP/c2_prune98_zero_rej.csv', 'C2: Prune 98 + Zero Rej')
eval_csv('phase2/V28_CHAMPIONSHIP/c3_oracle_rescue.csv', 'C3: Oracle Rescue (27+78)')
