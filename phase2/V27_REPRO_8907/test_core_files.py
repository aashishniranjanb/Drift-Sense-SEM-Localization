import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import glob
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')

pred_files = [
    'data/phase2_dev/v25_predictions.csv',
    'data/phase2_dev/v25_predictions_thresh.csv',
    'phase2/V27_FINAL/V25_BASELINE.csv',
    'phase2/V27_FINAL/v25_unthresholded.csv'
]

rows = []
for pf in pred_files:
    df_pred = pd.read_csv(pf)
    m = pd.merge(gt, df_pred, on='pair_id', suffixes=('_gt', '_pred'))
    if 'found' in m.columns and 'pred_found' not in m.columns:
        m['pred_found'] = m['found']
    m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])
    
    # Def 1: Official failure mode
    y1 = []
    for _, r in m.iterrows():
        gf, pf_val = r['gt_found'], r['pred_found']
        if gf == 1 and pf_val == 1:
            y1.append(1 if r['loc_err'] <= 5.0 else 0)
        elif gf == 0 and pf_val == 0:
            y1.append(1)
        else:
            y1.append(0)
    y1 = np.array(y1)
    
    # Def 2: Presence correctness (gt_found == pred_found)
    y2 = (m['gt_found'] == m['pred_found']).astype(int).values
    
    # Def 3: GT presence (gt_found == 1)
    y3 = m['gt_found'].values
    
    scores = m['score'].values
    
    sp1, _ = spearmanr(scores, y1)
    auc1 = roc_auc_score(y1, scores)
    auc2 = roc_auc_score(y2, scores)
    auc3 = roc_auc_score(y3, scores)
    ap1 = average_precision_score(y1, scores)
    ap3 = average_precision_score(y3, scores)
    
    # What about unthresholded / presence probability?
    rows.append({
        'file': pf,
        'Spearman_y1': round(sp1, 4),
        'AUC_y1 (Correctness)': round(auc1, 4),
        'AUC_y2 (Decision Match)': round(auc2, 4),
        'AUC_y3 (Presence GT)': round(auc3, 4),
        'AP_y1': round(ap1, 4),
        'AP_y3 (Presence AP)': round(ap3, 4)
    })

res_df = pd.DataFrame(rows)
print(res_df.to_string())
