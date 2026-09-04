import pandas as pd
import numpy as np

def run_diff(base_csv, exp_csv):
    df_base = pd.read_csv(base_csv)
    df_exp = pd.read_csv(exp_csv)
    m = pd.merge(df_base, df_exp, on="pair_id", suffixes=("_base", "_exp"))
    
    x_diff = np.sum(np.abs(m['x_base'] - m['x_exp']) > 1e-5)
    y_diff = np.sum(np.abs(m['y_base'] - m['y_exp']) > 1e-5)
    theta_diff = np.sum(np.abs(m['theta_base'] - m['theta_exp']) > 1e-5)
    scale_diff = np.sum(np.abs(m['scale_base'] - m['scale_exp']) > 1e-5)
    found_diff = np.sum(m['found_base'] != m['found_exp'])
    score_diff = np.sum(np.abs(m['score_base'] - m['score_exp']) > 1e-5)
    
    print("=== DIFF AUDIT ===")
    print(f"x changes: {x_diff}")
    print(f"y changes: {y_diff}")
    print(f"theta changes: {theta_diff}")
    print(f"found changes: {found_diff}")
    print(f"scale changes: {scale_diff}")
    print(f"score changes: {score_diff}")

    # calculate pose MAE for scale
    gt = pd.read_csv('data/phase2_dev/pairs.csv')
    mg = pd.merge(m, gt, on="pair_id")
    
    # We only care about scale MAE for found=1
    mg_f1 = mg[(mg['gt_found'] == 1) & (mg['found_base'] == 1)]
    
    scale_mae_base = np.mean(np.abs(mg_f1['scale_base'] - mg_f1['gt_scale']))
    scale_mae_exp = np.mean(np.abs(mg_f1['scale_exp'] - mg_f1['gt_scale']))
    
    print("\n=== SCALE MAE ===")
    print(f"Baseline Scale MAE: {scale_mae_base:.4f}")
    print(f"Scale-Only Scale MAE: {scale_mae_exp:.4f}")

run_diff('FINAL_SUBMISSION/predictions.csv', 'FINAL_SUBMISSION/validation/scale_only.csv')
