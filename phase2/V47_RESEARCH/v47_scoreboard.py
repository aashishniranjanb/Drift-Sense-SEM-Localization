import pandas as pd
import numpy as np
import os

df = pd.read_csv('phase2/V47_RESEARCH/v47_candidate_cache/features_oof.csv')

def eval_loc(cx, cy, gt_x, gt_y):
    if np.isnan(gt_x) or np.isnan(gt_y): return 99
    return np.hypot(cx - gt_x, cy - gt_y)

audit_rows = []

def eval_gate(df, prob_col, thresh_absent, thresh_present):
    rescued = 0
    broken = 0
    new_fp = 0
    
    for _, r in df.iterrows():
        v25_found = r['v25_found']
        gt_found = r['gt_found']
        prob = r[prob_col]
        cons = r['v46_consensus']
        v46_score = r['v46_score']
        
        # Calculate v25 score from features (approximate)
        # We don't have exactly v25_rescue_score, but we can assume v46_score > v25_score+0.10 is mostly true for rescues.
        # Actually in V46-D we found 17 cases. Let's just use the prob!
        
        override = False
        if v25_found == 0:
            if prob >= thresh_absent and cons >= 3: override = True
        else:
            if r['v25_ml_score'] < 0.95:
                if prob >= thresh_present and cons >= 3: override = True
                
        if override:
            d_v46 = eval_loc(r['v46_cx'], r['v46_cy'], r['gt_x'], r['gt_y'])
            d_v25 = eval_loc(r['v25_cx'], r['v25_cy'], r['gt_x'], r['gt_y'])
            
            if gt_found == 1:
                v25_corr = (d_v25 <= 5.0 and v25_found == 1)
                v46_corr = (d_v46 <= 5.0)
                if v46_corr and not v25_corr: rescued += 1
                elif not v46_corr and v25_corr: broken += 1
            else:
                if v25_found == 0: new_fp += 1
                
    return rescued, broken, new_fp

models = ['oof_lr', 'oof_hgb2', 'oof_hgb3', 'oof_hand']
thresholds = np.linspace(0.1, 0.95, 18)

print("=== V47 SCOREBOARD ===")
print(f"{'Model':<10} {'T_absent':<8} {'T_present':<9} {'Rescued':<7} {'Broken':<7} {'New FP':<7}")
print("-" * 55)

best_results = []

for m in models:
    best_res = -1
    best_stats = None
    best_t = None
    for t_abs in thresholds:
        for t_pres in thresholds:
            res, brk, fp = eval_gate(df, m, t_abs, t_pres)
            if brk <= 1 and fp <= 1:
                if res > best_res:
                    best_res = res
                    best_stats = (res, brk, fp)
                    best_t = (t_abs, t_pres)
                    
    if best_stats:
        print(f"{m:<10} {best_t[0]:<8.2f} {best_t[1]:<9.2f} {best_stats[0]:<7} {best_stats[1]:<7} {best_stats[2]:<7}")
    else:
        print(f"{m:<10} No safe threshold found")
        
