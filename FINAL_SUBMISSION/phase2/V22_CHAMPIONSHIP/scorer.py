import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import os
import argparse

def compute_competition_score(merged_df, runtime_median=2.0):
    """
    merged_df: GT + predictions merged on pair_id.
    """
    df = merged_df.copy()
    
    # 1. Localization (40 pts)
    # Euclidean distance from predicted x,y to gt_x,gt_y (PRESENT only, pred_found=1)
    df['loc_err'] = np.where(
        (df['pred_found'] == 1) & (df['gt_found'] == 1),
        np.sqrt((df['x'] - df['gt_x'])**2 + (df['y'] - df['gt_y'])**2),
        np.nan
    )
    
    def loc_credit(err):
        if pd.isna(err): return 0.0
        if err <= 1.0: return 1.0
        if err <= 2.0: return 0.8
        if err <= 3.0: return 0.6
        if err <= 5.0: return 0.4
        return 0.0

    df['loc_credit'] = df['loc_err'].apply(loc_credit)
    
    set_a_mask = df['set_type'] == 'SetA'
    set_b_mask = df['set_type'] == 'SetB'
    set_c_mask = df['set_type'] == 'SetC' # Assuming SetC is also present, weight is 0 for loc if it doesn't have GT?
    
    # Set A weight 0.45, Set B weight 0.55
    # Wait, 180 total pairs, 70 SetA, 70 SetB, 40 SetC. SetC might not be included in loc score.
    # Total max score is 40. We will average over SetA and SetB separately, then combine.
    loc_a = df.loc[set_a_mask & (df['gt_found'] == 1), 'loc_credit'].mean()
    loc_b = df.loc[set_b_mask & (df['gt_found'] == 1), 'loc_credit'].mean()
    if pd.isna(loc_a): loc_a = 0
    if pd.isna(loc_b): loc_b = 0
    
    localization_score = 40.0 * (0.45 * loc_a + 0.55 * loc_b)
    
    # 2. Pose (20 pts)
    # Pose only scored when localization receives credit (loc_err <= 5px and pred_found=1)
    df['pose_eligible'] = (df['loc_err'] <= 5.0) & (df['pred_found'] == 1) & (df['gt_found'] == 1)
    
    df['scale_err'] = np.abs(df['scale'] - df['gt_scale']) / df['gt_scale'] * 100
    df['rot_err'] = np.abs(df['theta'] - df['gt_theta'])
    # wrap rotation error to 0-180 just in case
    df['rot_err'] = np.minimum(df['rot_err'], 360 - df['rot_err'])
    
    def scale_credit(err):
        if pd.isna(err): return 0.0
        if err <= 1.0: return 1.0
        if err <= 2.0: return 0.75
        if err <= 5.0: return 0.5
        return 0.0
        
    def rot_credit(err):
        if pd.isna(err): return 0.0
        if err <= 0.25: return 1.0
        if err <= 0.5: return 0.75
        if err <= 1.0: return 0.5
        return 0.0

    df['scale_score'] = df['scale_err'].apply(scale_credit)
    df['rot_score'] = df['rot_err'].apply(rot_credit)
    
    # Pose score is only where eligible
    valid_pose_a = df.loc[set_a_mask & df['pose_eligible']]
    valid_pose_b = df.loc[set_b_mask & df['pose_eligible']]
    
    scale_a = valid_pose_a['scale_score'].mean() if not valid_pose_a.empty else 0
    scale_b = valid_pose_b['scale_score'].mean() if not valid_pose_b.empty else 0
    rot_a = valid_pose_a['rot_score'].mean() if not valid_pose_a.empty else 0
    rot_b = valid_pose_b['rot_score'].mean() if not valid_pose_b.empty else 0
    
    if pd.isna(scale_a): scale_a = 0
    if pd.isna(scale_b): scale_b = 0
    if pd.isna(rot_a): rot_a = 0
    if pd.isna(rot_b): rot_b = 0
    
    # Combining scale and rotation (each out of 10)
    scale_pts = 10.0 * (0.45 * scale_a + 0.55 * scale_b)
    rot_pts = 10.0 * (0.45 * rot_a + 0.55 * rot_b)
    pose_score = scale_pts + rot_pts
    
    # 3. Rejection (15 pts)
    # F1 across all 180 cases where positive class = found=0 (ABSENT)
    y_true_rej = (df['gt_found'] == 0).astype(int)
    y_pred_rej = (df['pred_found'] == 0).astype(int)
    rejection_f1 = f1_score(y_true_rej, y_pred_rej, zero_division=0)
    rejection_score = 15.0 * rejection_f1
    
    # 4. Calibration (10 pts)
    # AUC of (score) column against per-pair correctness
    # Correctness = 1 if (loc_err <= 5px AND pred_found == gt_found) else 0 (also for absent it is pred_found == gt_found)
    df['correctness'] = 0
    # True positives
    df.loc[(df['gt_found'] == 1) & (df['pred_found'] == 1) & (df['loc_err'] <= 5.0), 'correctness'] = 1
    # True negatives
    df.loc[(df['gt_found'] == 0) & (df['pred_found'] == 0), 'correctness'] = 1
    
    try:
        calibration_auc = roc_auc_score(df['correctness'], df['score'])
    except ValueError:
        calibration_auc = 0.5 # If only one class
    calibration_score = 10.0 * calibration_auc
    
    # 5. Efficiency (5 pts)
    # Median runtime <= 5s => full credit
    efficiency_score = 5.0 if runtime_median <= 5.0 else max(0, 5.0 - (runtime_median - 5.0))
    
    # 6. Generator/Docs (10 pts)
    docs_score = 10.0
    
    total_score = localization_score + pose_score + rejection_score + calibration_score + efficiency_score + docs_score
    
    # Other metrics of interest
    present_recall = ((df['gt_found'] == 1) & (df['pred_found'] == 1)).sum() / (df['gt_found'] == 1).sum()
    
    setb_loc_leq_5 = df.loc[set_b_mask & (df['gt_found'] == 1), 'loc_err'] <= 5.0
    setb_loc_acc = setb_loc_leq_5.mean()
    
    return {
        'Total Score': total_score,
        'Localization (40)': localization_score,
        'Pose (20)': pose_score,
        'Rejection (15)': rejection_score,
        'Calibration (10)': calibration_score,
        'Efficiency (5)': efficiency_score,
        'Docs (10)': docs_score,
        'Rejection F1': rejection_f1,
        'Calibration AUC': calibration_auc,
        'PRESENT recall': present_recall,
        'Set B <= 5px': setb_loc_acc,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='data/phase2_dev/pairs.csv')
    parser.add_argument('--pred', type=str, default='data/phase2_dev/predictions.csv')
    parser.add_argument('--out', type=str, default='phase2/V22_CHAMPIONSHIP/results/V22_A_control.csv')
    args = parser.parse_args()
    
    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)
    # Fix pred columns to merge cleanly
    pred = pred.rename(columns={'found': 'pred_found'})
    
    merged = pd.merge(gt, pred, on='pair_id', how='inner')
    
    scores = compute_competition_score(merged)
    
    import json
    print(json.dumps(scores, indent=2))
    
    pd.DataFrame([scores]).to_csv(args.out, index=False)
