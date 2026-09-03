import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

def calculate_exact_scores(gt_csv, pred_csv):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)
    merged = pd.merge(gt, pred, on="pair_id", suffixes=("_gt", "_pred"))
    
    # 1. Localization
    set_a = merged[(merged['set_type'] == 'SetA') & (merged['gt_found'] == 1)]
    set_b = merged[(merged['set_type'] == 'SetB') & (merged['gt_found'] == 1)]
    
    def loc_pct(df):
        df_found = df[df['found'] == 1].copy()
        if len(df_found) == 0: return 0.0
        df_found['err'] = np.hypot(df_found['x'] - df_found['gt_x'], df_found['y'] - df_found['gt_y'])
        return np.mean(df_found['err'] <= 5.0) * 100.0 * (len(df_found) / len(df)) # Wait, is it over all present pairs?
        
    def loc_pct_official(df):
        # The benchmark script does:
        # localized = present_gt[present_gt["pred_found"] == 1]
        # errs = localized["loc_err"].values
        # le_5 = np.mean(errs <= 5.0) * 100.0 if len(errs) > 0 else 0.0
        # Wait, if we reject a pair, it's NOT counted in the denominator for le_5 in benchmark_phase2.py!
        # "localized = present_gt[present_gt["pred_found"] == 1]" -> errs = localized.loc_err -> mean.
        # This means le_5 is the accuracy OF THE ACCEPTED PREDICTIONS!
        localized = df[df['found'] == 1].copy()
        if len(localized) == 0: return 0.0
        localized['err'] = np.hypot(localized['x'] - localized['gt_x'], localized['y'] - localized['gt_y'])
        return np.mean(localized['err'] <= 5.0) * 100.0
        
    a_le5 = loc_pct_official(set_a)
    b_le5 = loc_pct_official(set_b)
    loc_score_pct = 0.45 * a_le5 + 0.55 * b_le5
    loc_points = loc_score_pct * 0.40
    
    # 2. Rejection F1
    # where found == 0 is the positive class
    tp_rej = np.sum((merged["gt_found"] == 0) & (merged["found"] == 0))
    fp_rej = np.sum((merged["gt_found"] == 1) & (merged["found"] == 0))
    fn_rej = np.sum((merged["gt_found"] == 0) & (merged["found"] == 1))
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    rej_points = f1_rej * 15.0
    
    # 3. Calibration
    # The user said: "Optimize the actual Phase 2 calibration metric. Do NOT use Spearman as a replacement".
    # Wait, in the V25 prompt the user mentioned "Calibration AUC" earlier: "Exp E: Construct an OOF confidence score representing P(final prediction is correct) to maximize Calibration AUC."
    # Let's calculate AUC.
    correctness = []
    for _, row in merged.iterrows():
        gt_f = row['gt_found']
        pr_f = row['found']
        if gt_f == 1 and pr_f == 1:
            err = np.hypot(row['x'] - row['gt_x'], row['y'] - row['gt_y'])
            correctness.append(1 if err <= 5.0 else 0)
        elif gt_f == 0 and pr_f == 0:
            correctness.append(1)
        else:
            correctness.append(0)
            
    auc = roc_auc_score(correctness, merged['score']) if len(set(correctness)) > 1 else 0.0
    spearman_corr, _ = spearmanr(merged['score'], correctness)
    
    return {
        'loc_points': loc_points,
        'rej_points': rej_points,
        'auc': auc,
        'spearman': spearman_corr
    }

print(calculate_exact_scores('data/phase2_dev/pairs.csv', 'data/phase2_dev/v25_predictions_thresh.csv'))
