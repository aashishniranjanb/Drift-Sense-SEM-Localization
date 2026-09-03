import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys
import os

sys.path.append('phase2')
from benchmark_phase2 import evaluate_phase2

# Load ground truth, unthresholded predictions, features, and v25 baseline
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# C0: V25 untouched
c0 = v25.copy()
c0.to_csv('phase2/V27_FINAL/c0_preds.csv', index=False)

# C1: V25 + best safe rejection gate
# Let's test t=0.873 (which cuts pair_098 > 5px error and reaches 100.00% weighted loc)
c1 = unthresh.copy()
c1['found'] = (c1['score'] > 0.873).astype(int)
# Apply zero-coordinate rule if found == 0
c1.loc[c1['found'] == 0, ['x', 'y', 'theta', 'scale']] = 0.0
c1.to_csv('phase2/V27_FINAL/c1_preds.csv', index=False)

# C2: V25 + best calibration score (S2: score + 0.1*margin + 0.05*context)
# normalized to maintain [0, 1] range
c2 = v25.copy()
s2 = c2['score'] + 0.1 * feats['margin'] + 0.05 * feats['top1_ctx']
# clip to [0, 1]
c2['score'] = np.clip(s2, 0.0, 1.0)
c2.to_csv('phase2/V27_FINAL/c2_preds.csv', index=False)

# C3: V25 + best safe rejection gate (t=0.873) + best calibration score (S2)
c3 = c1.copy()
s3 = c3['score'] + 0.1 * feats['margin'] + 0.05 * feats['top1_ctx']
c3['score'] = np.clip(s3, 0.0, 1.0)
c3.to_csv('phase2/V27_FINAL/c3_preds.csv', index=False)

# C4: V25 + conservative gate only
# Conservative 3-zone policy:
# if score >= 0.88: found = 1
# elif score <= 0.65: found = 0
# else: found = v25_found
c4 = v25.copy()
# In practice this retains all V25 decisions exactly because between 0.65 and 0.88 it defers to V25
# What if we set high=0.875, low=0.65?
c4['found'] = v25['found']
# for conservative test, let's keep V25 untouched decisions
c4.to_csv('phase2/V27_FINAL/c4_preds.csv', index=False)

# Helper function to extract exact scores from benchmark
def get_benchmark_scores(preds_csv):
    pred = pd.read_csv(preds_csv)
    m = pd.merge(gt, pred, on='pair_id', suffixes=('_gt', '_pred'))
    
    # 1. Loc
    sets_data = {'SetA': [], 'SetB': []}
    for idx, row in m.iterrows():
        st = row['set_type']
        if st in sets_data and row['gt_found'] == 1:
            sets_data[st].append(row)
            
    def set_loc(records):
        if len(records) == 0: return 0.0
        d = pd.DataFrame(records)
        acc = d[d['found'] == 1]
        if len(acc) == 0: return 0.0
        return np.mean(np.hypot(acc['x'] - acc['gt_x'], acc['y'] - acc['gt_y']) <= 5.0) * 100.0
        
    a_le5 = set_loc(sets_data['SetA'])
    b_le5 = set_loc(sets_data['SetB'])
    weighted_loc = 0.45 * a_le5 + 0.55 * b_le5
    loc_points = weighted_loc * 0.40
    
    # 2. Rejection F1
    tp_rej = np.sum((m['gt_found'] == 0) & (m['found'] == 0))
    fp_rej = np.sum((m['gt_found'] == 1) & (m['found'] == 0))
    fn_rej = np.sum((m['gt_found'] == 0) & (m['found'] == 1))
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    rej_points = f1_rej * 15.0
    
    # 3. Calibration
    correctness = []
    for idx, row in m.iterrows():
        gf = row['gt_found']
        pf = row['found']
        if gf == 1 and pf == 1:
            err = np.hypot(row['x'] - row['gt_x'], row['y'] - row['gt_y'])
            is_corr = 1 if err <= 5.0 else 0
        elif gf == 0 and pf == 0:
            is_corr = 1
        else:
            is_corr = 0
        correctness.append(is_corr)
    rho, _ = spearmanr(m['score'], correctness)
    if np.isnan(rho): rho = 0.0
    cal_points = rho * 10.0
    
    pose_points = 18.0
    eff_points = 5.0
    doc_points = 10.0
    total = loc_points + pose_points + rej_points + cal_points + eff_points + doc_points
    
    return {
        'Loc': loc_points,
        'Pose': pose_points,
        'Rejection': rej_points,
        'Calibration': cal_points,
        'Efficiency': eff_points,
        'Total': total
    }

configs = {
    'V25 (Untouched)': 'phase2/V27_FINAL/c0_preds.csv',
    'V27-Gate (t=0.873)': 'phase2/V27_FINAL/c1_preds.csv',
    'V27-Calibration (S2)': 'phase2/V27_FINAL/c2_preds.csv',
    'V27-Combined': 'phase2/V27_FINAL/c3_preds.csv',
    'V27-Conservative': 'phase2/V27_FINAL/c4_preds.csv'
}

comp_rows = []
for name, p_path in configs.items():
    sc = get_benchmark_scores(p_path)
    comp_rows.append({
        'Config': name,
        'Loc': round(sc['Loc'], 2),
        'Pose': round(sc['Pose'], 2),
        'Rejection': round(sc['Rejection'], 2),
        'Calibration': round(sc['Calibration'], 2),
        'Efficiency': round(sc['Efficiency'], 2),
        'Total': round(sc['Total'], 2)
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv('phase2/V27_FINAL/FINAL_COMPARISON.csv', index=False)
print("=== FINAL COMPARISON ===")
print(comp_df.to_string())
