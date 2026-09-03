import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def run_audit():
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    preds = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
    tax = pd.read_csv('data/phase2_dev/failure_taxonomy.csv')
    
    # Merge
    df = pairs.merge(preds, on='pair_id').merge(tax[['pair_id', 'loc_err', 'scale_err', 'theta_err', 'failure_mode']], on='pair_id', how='left')
    
    # 1. Localization
    set_a = df[(df['set_type'] == 'SetA') & (df['found'] == 1) & (df['gt_found'] == 1)]
    set_b = df[(df['set_type'] == 'SetB') & (df['found'] == 1) & (df['gt_found'] == 1)]
    le5_a = np.mean(set_a['loc_err'] <= 5.0) if len(set_a) > 0 else 0
    le5_b = np.mean(set_b['loc_err'] <= 5.0) if len(set_b) > 0 else 0
    weighted_loc = 0.45 * le5_a + 0.55 * le5_b
    loc_points = weighted_loc * 40.0
    loc_loss = 40.0 - loc_points
    
    # 2. Rejection
    tp_rej = np.sum((df['gt_found'] == 0) & (df['found'] == 0))
    fp_rej = np.sum((df['gt_found'] == 1) & (df['found'] == 0))
    fn_rej = np.sum((df['gt_found'] == 0) & (df['found'] == 1))
    
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0
    rej_points = f1_rej * 15.0
    rej_loss = 15.0 - rej_points
    
    false_reject_loss = 0
    false_accept_loss = 0
    if (fp_rej + fn_rej) > 0:
        # Re-calc F1 if we had NO false positives
        f1_perfect_fp = 2 * (tp_rej / (tp_rej)) * rec_rej / ((tp_rej / (tp_rej)) + rec_rej) if rec_rej > 0 else 0
        f1_perfect_fn = 2 * prec_rej * (tp_rej / tp_rej) / (prec_rej + (tp_rej / tp_rej)) if prec_rej > 0 else 0
        
        fp_impact = (f1_perfect_fp - f1_rej) * 15.0
        fn_impact = (f1_perfect_fn - f1_rej) * 15.0
        total_impact = fp_impact + fn_impact
        if total_impact > 0:
            false_reject_loss = rej_loss * (fp_impact / total_impact)
            false_accept_loss = rej_loss * (fn_impact / total_impact)
    
    # 3. Calibration
    scores = df['score'].values
    corr = np.array([1 if (fm in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"]) else 0 for fm in df['failure_mode']])
    rho, _ = spearmanr(scores, corr)
    if np.isnan(rho): rho = 0
    cal_points = rho * 10.0
    cal_loss = 10.0 - cal_points
    
    pose_loss = 2.0
    
    total = loc_points + (20.0 - pose_loss) + rej_points + cal_points + 5 + 10
    
    print("\nScore-Loss Waterfall")
    print("100")
    print(f" ?")
    print(f" ??? localization loss: {loc_loss:.2f}")
    print(f" ??? pose loss:         {pose_loss:.2f}")
    print(f" ??? false reject loss: {false_reject_loss:.2f} (FP in rejection: {fp_rej} cases)")
    print(f" ??? false accept loss: {false_accept_loss:.2f} (FN in rejection: {fn_rej} cases)")
    print(f" ??? calibration loss:  {cal_loss:.2f}")
    print(f" ?")
    print(f" ??? total remaining:   {total:.2f}")
    
run_audit()
