import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from scipy.stats import spearmanr, kendalltau, pearsonr
import glob
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')

# Find all prediction files in the repository
pred_files = [
    'data/phase2_dev/v25_predictions.csv',
    'data/phase2_dev/v25_predictions_thresh.csv',
    'phase2/V27_FINAL/V25_BASELINE.csv',
    'phase2/V27_FINAL/v25_unthresholded.csv',
    'data/phase2_dev/predictions.csv'
]

# Add any other prediction files in phase2
for f in glob.glob('phase2/**/*.csv', recursive=True):
    if 'pred' in f.lower() or 'baseline' in f.lower():
        if f not in pred_files:
            pred_files.append(f)

print(f"Testing {len(pred_files)} prediction files...")

results = []

for pf in pred_files:
    try:
        df_pred = pd.read_csv(pf)
        if 'pair_id' not in df_pred.columns or 'score' not in df_pred.columns:
            continue
        if len(df_pred) != 180:
            continue
            
        m = pd.merge(gt, df_pred, on='pair_id', suffixes=('_gt', '_pred'))
        if 'found' in m.columns and 'pred_found' not in m.columns:
            m['pred_found'] = m['found']
            
        # Calculate localization error
        m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])
        
        # Define various candidate definitions of correctness y:
        # Def 1: Official benchmark failure mode definition
        # is_corr = 1 if failure_mode in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"]
        y1 = []
        for _, r in m.iterrows():
            gf, pf = r['gt_found'], r['pred_found']
            if gf == 1 and pf == 1:
                y1.append(1 if r['loc_err'] <= 5.0 else 0)
            elif gf == 0 and pf == 0:
                y1.append(1)
            else:
                y1.append(0)
        y1 = np.array(y1)
        
        # Def 2: Presence correctness (gt_found == pred_found)
        y2 = (m['gt_found'] == m['pred_found']).astype(int).values
        
        # Def 3: Target ground truth presence (gt_found == 1)
        y3 = m['gt_found'].values
        
        # Def 4: Localization <= 1px or rejection success
        y4 = []
        for _, r in m.iterrows():
            gf, pf = r['gt_found'], r['pred_found']
            if gf == 1 and pf == 1:
                y4.append(1 if r['loc_err'] <= 1.0 else 0)
            elif gf == 0 and pf == 0:
                y4.append(1)
            else:
                y4.append(0)
        y4 = np.array(y4)
        
        # Def 5: Correctness among accepted only (len = 80 for v25)
        m_acc = m[m['pred_found'] == 1]
        y_acc = (m_acc['loc_err'] <= 5.0).astype(int).values if len(m_acc) > 0 else np.array([])
        
        # Def 6: Correctness among present only (len = 140)
        m_pres = m[m['gt_found'] == 1]
        y_pres = ((m_pres['pred_found'] == 1) & (m_pres['loc_err'] <= 5.0)).astype(int).values
        
        scores = m['score'].values
        
        # Compute metrics across definitions
        # 1. Spearman
        sp1, _ = spearmanr(scores, y1)
        sp2, _ = spearmanr(scores, y2)
        sp3, _ = spearmanr(scores, y3)
        sp4, _ = spearmanr(scores, y4)
        
        # 2. ROC-AUC
        auc1 = roc_auc_score(y1, scores) if len(np.unique(y1)) > 1 else np.nan
        auc2 = roc_auc_score(y2, scores) if len(np.unique(y2)) > 1 else np.nan
        auc3 = roc_auc_score(y3, scores) if len(np.unique(y3)) > 1 else np.nan
        auc4 = roc_auc_score(y4, scores) if len(np.unique(y4)) > 1 else np.nan
        auc_acc = roc_auc_score(y_acc, m_acc['score'].values) if len(np.unique(y_acc)) > 1 else np.nan
        auc_pres = roc_auc_score(y_pres, m_pres['score'].values) if len(np.unique(y_pres)) > 1 else np.nan
        
        # 3. Average Precision
        ap1 = average_precision_score(y1, scores)
        ap2 = average_precision_score(y2, scores)
        ap3 = average_precision_score(y3, scores)
        
        # 4. Pearson r
        pe1, _ = pearsonr(scores, y1)
        pe2, _ = pearsonr(scores, y2)
        pe3, _ = pearsonr(scores, y3)
        
        results.append({
            'file': os.path.basename(pf),
            'sp_y1(bench)': sp1,
            'auc_y1(bench)': auc1,
            'sp_y2(presence)': sp2,
            'auc_y2(presence)': auc2,
            'sp_y3(gt_pres)': sp3,
            'auc_y3(gt_pres)': auc3,
            'auc_pres(140)': auc_pres,
            'ap_y1': ap1,
            'ap_y3': ap3,
            'pe_y1': pe1
        })
    except Exception as e:
        pass

res_df = pd.DataFrame(results)
print(res_df.to_string())
