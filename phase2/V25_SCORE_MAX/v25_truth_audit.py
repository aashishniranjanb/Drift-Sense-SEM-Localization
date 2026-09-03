import pandas as pd
import numpy as np
import subprocess
import re
import os

def parse_score(output_text):
    loc = 0.0
    rej = 0.0
    cal = 0.0
    
    for line in output_text.split('\n'):
        if 'OFFICIAL WEIGHTED LOC SCORE' in line:
            m = re.search(r'([\d\.]+)%', line)
            if m: loc = float(m.group(1)) * 40.0 / 100.0
        elif 'Set C Rejection F1 Score:' in line:
            m = re.search(r'([\d\.]+)', line.split(':')[-1])
            if m: rej = float(m.group(1)) * 15.0
        elif 'Spearman Rank Correlation (rho):' in line:
            m = re.search(r'([\d\.]+)', line.split(':')[-1])
            if m: cal = float(m.group(1)) * 10.0
            
    # Assuming pose is constant ~18.0, efficiency=5, docs=10 for delta purposes
    return loc, rej, cal

def main():
    print("STEP 0 - TRUTH TABLE VERIFICATION")
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    preds_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
    df = pairs.merge(preds_thresh, on='pair_id')
    
    actual_present = df['set_type'].isin(['SetA', 'SetB']).astype(int)
    pred_present = df['found']
    
    TP = np.sum((actual_present == 1) & (pred_present == 1))
    FN = np.sum((actual_present == 1) & (pred_present == 0))
    FP = np.sum((actual_present == 0) & (pred_present == 1))
    TN = np.sum((actual_present == 0) & (pred_present == 0))
    
    print(f"TP (Present|Present): {TP}")
    print(f"FN (Present|Absent):  {FN}")
    print(f"FP (Absent|Present):  {FP}")
    print(f"TN (Absent|Absent):   {TN}")
    print(f"TP+FN = {TP+FN} (Expected 140)")
    print(f"FP+TN = {FP+TN} (Expected 40)")
    
    # Explain Rejection F1
    # Target class = ABSENT (0)
    tp_rej = TN
    fp_rej = FN
    fn_rej = FP
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0
    print(f"\nRejection F1 (Positive Class = ABSENT):")
    print(f"TP_rej = {tp_rej}, FP_rej = {fp_rej}, FN_rej = {fn_rej}")
    print(f"F1 = 2 * {tp_rej} / (2*{tp_rej} + {fp_rej} + {fn_rej}) = {f1_rej:.4f}\n")
    
    print("STEP 1 - COUNTERFACTUAL SCORE AUDIT")
    raw_preds = pd.read_csv('data/phase2_dev/v25_predictions.csv')
    
    # Get base score
    res = subprocess.run(['python', 'phase2/benchmark_phase2.py', '--input-csv', 'data/phase2_dev/pairs.csv', '--predictions-csv', 'data/phase2_dev/v25_predictions_thresh.csv'], capture_output=True, text=True)
    base_loc, base_rej, base_cal = parse_score(res.stdout)
    base_total = base_loc + base_rej + base_cal
    
    print(f"Base Loc: {base_loc:.2f}, Rej: {base_rej:.2f}, Cal: {base_cal:.2f} | Total: {base_total:.2f}")
    
    false_reject_ids = df[(actual_present == 1) & (pred_present == 0)]['pair_id'].values
    print(f"Analyzing {len(false_reject_ids)} False Rejects...")
    
    results = []
    for pid in false_reject_ids:
        # Create counterfactual CSV
        cf_df = preds_thresh.copy()
        idx = cf_df[cf_df['pair_id'] == pid].index[0]
        raw_row = raw_preds[raw_preds['pair_id'] == pid].iloc[0]
        
        # Restore raw predictions
        for c in ['x', 'y', 'theta', 'scale']:
            cf_df.loc[idx, c] = raw_row[c]
        cf_df.loc[idx, 'found'] = 1
        
        cf_path = 'data/phase2_dev/temp_cf.csv'
        cf_df.to_csv(cf_path, index=False)
        
        res = subprocess.run(['python', 'phase2/benchmark_phase2.py', '--input-csv', 'data/phase2_dev/pairs.csv', '--predictions-csv', cf_path], capture_output=True, text=True)
        cf_loc, cf_rej, cf_cal = parse_score(res.stdout)
        cf_total = cf_loc + cf_rej + cf_cal
        
        delta = cf_total - base_total
        
        # We also want to know if it was a good localization. 
        # Check raw failure mode for this pair
        # We can approximate by seeing if delta_loc > 0
        delta_loc = cf_loc - base_loc
        results.append({'pair_id': pid, 'delta_total': delta, 'delta_loc': delta_loc, 'delta_rej': cf_rej - base_rej, 'delta_cal': cf_cal - base_cal})
        
    res_df = pd.DataFrame(results).sort_values('delta_total', ascending=False)
    print("\nTop 15 Highest ROI False Rejects to Rescue:")
    print(res_df.head(15).to_string(index=False))
    
    print("\nSummary of Deltas:")
    print(f"Cases with Positive Delta (>0.01): {len(res_df[res_df['delta_total'] > 0.01])}")
    print(f"Cases with Negative Delta (<-0.01): {len(res_df[res_df['delta_total'] < -0.01])}")
    print(f"Cases with Neutral Delta: {len(res_df[(res_df['delta_total'] >= -0.01) & (res_df['delta_total'] <= 0.01)])}")

if __name__ == '__main__':
    main()
