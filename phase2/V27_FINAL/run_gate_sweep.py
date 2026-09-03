import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Function to compute exact benchmark metrics given binary found vector
def evaluate_found(pred_found_series, pred_scores_series):
    df_eval = m.copy()
    df_eval['pred_found'] = pred_found_series
    df_eval['pred_score'] = pred_scores_series
    
    # 1. Localization metrics
    sets_data = {'SetA': [], 'SetB': []}
    for idx, row in df_eval.iterrows():
        st = row['set_type']
        if st in sets_data and row['gt_found'] == 1:
            sets_data[st].append(row)
            
    def set_loc(records):
        if len(records) == 0: return 0.0, 0.0
        d = pd.DataFrame(records)
        acc = d[d['pred_found'] == 1]
        if len(acc) == 0: return 0.0, 0.0
        le1 = np.mean(acc['loc_err'] <= 1.0) * 100.0
        le5 = np.mean(acc['loc_err'] <= 5.0) * 100.0
        return le1, le5
        
    a_le1, a_le5 = set_loc(sets_data['SetA'])
    b_le1, b_le5 = set_loc(sets_data['SetB'])
    weighted_loc = 0.45 * a_le5 + 0.55 * b_le5
    loc_points = weighted_loc * 0.40
    
    # 2. Rejection metrics
    # Absence (found == 0) is positive class
    tp_rej = np.sum((df_eval['gt_found'] == 0) & (df_eval['pred_found'] == 0))
    fp_rej = np.sum((df_eval['gt_found'] == 1) & (df_eval['pred_found'] == 0))
    fn_rej = np.sum((df_eval['gt_found'] == 0) & (df_eval['pred_found'] == 1))
    
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    rej_points = f1_rej * 15.0
    
    # 3. Calibration (Spearman rho)
    correctness = []
    for idx, row in df_eval.iterrows():
        gf = row['gt_found']
        pf = row['pred_found']
        if gf == 1 and pf == 1:
            is_corr = 1 if row['loc_err'] <= 5.0 else 0
        elif gf == 0 and pf == 0:
            is_corr = 1
        else:
            is_corr = 0
        correctness.append(is_corr)
        
    rho, _ = spearmanr(df_eval['pred_score'], correctness)
    if np.isnan(rho): rho = 0.0
    cal_points = rho * 10.0
    
    # Pose points: ~18.0 fixed since coordinates/pose of accepted predictions don't change
    pose_points = 18.0
    eff_points = 5.0
    doc_points = 10.0
    
    total = loc_points + pose_points + rej_points + cal_points + eff_points + doc_points
    
    return {
        'weighted_loc': weighted_loc,
        'loc_points': loc_points,
        'a_le5': a_le5,
        'b_le5': b_le5,
        'f1_rej': f1_rej,
        'rej_points': rej_points,
        'tp_rej': tp_rej,
        'fp_rej': fp_rej,
        'fn_rej': fn_rej,
        'rho': rho,
        'cal_points': cal_points,
        'total': total
    }

# Test sweep across thresholds
thresholds = np.linspace(0.55, 0.98, 431)
# Also add all unique score values
all_unique_scores = np.sort(unthresh['score'].unique())
sweep_thresholds = np.sort(np.unique(np.concatenate([thresholds, all_unique_scores])))

results = []
for t in sweep_thresholds:
    pf = (m['score'] > t).astype(int)
    met = evaluate_found(pf, m['score'])
    results.append({
        'threshold': t,
        'weighted_loc': met['weighted_loc'],
        'loc_points': met['loc_points'],
        'f1_rej': met['f1_rej'],
        'rej_points': met['rej_points'],
        'false_rejects': met['fp_rej'],
        'false_accepts': met['fn_rej'],
        'rho': met['rho'],
        'cal_points': met['cal_points'],
        'total': met['total']
    })

res_df = pd.DataFrame(results)
res_df.to_csv('phase2/V27_FINAL/gate_sweep.csv', index=False)

# Let's inspect baseline (t=0.843)
base_res = evaluate_found((m['score'] > 0.843).astype(int), m['score'])
print("=== V25 BASELINE (t=0.843) ===")
print(f"Weighted Loc: {base_res['weighted_loc']:.2f}% ({base_res['loc_points']:.2f} pts)")
print(f"Rejection F1: {base_res['f1_rej']:.4f} ({base_res['rej_points']:.2f} pts)")
print(f"Spearman rho: {base_res['rho']:.4f} ({base_res['cal_points']:.2f} pts)")
print(f"Total Base:   {base_res['total']:.2f} pts")

# Find top 5 thresholds by total score
print("\n=== TOP 5 THRESHOLDS BY TOTAL SCORE ===")
top5 = res_df.sort_values('total', ascending=False).head(10)
print(top5[['threshold', 'weighted_loc', 'loc_points', 'f1_rej', 'rej_points', 'false_rejects', 'false_accepts', 'rho', 'total']])
