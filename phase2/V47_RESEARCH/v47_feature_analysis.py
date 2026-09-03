import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

df = pd.read_csv('phase2/V47_RESEARCH/v47_candidate_cache/features.csv')
# We care about separating Pop B (Gold rescues) from Pop C (Absent False Candidates)
df_eval = df[df['pop'].isin(['B', 'C'])].copy()

# Label: 1 if B, 0 if C
df_eval['label'] = (df_eval['pop'] == 'B').astype(int)

features = [
    'ncc', 'grad', 'ctx', 'phase',
    'ncc_pct', 'grad_pct', 'ctx_pct', 'phase_pct',
    'prom5_ncc', 'prom10_ncc', 'prom20_ncc',
    'prom5_grad', 'prom10_grad',
    'z5_ncc', 'z10_ncc',
    'comp10', 'comp20', 'comp40', 'd1', 'd2',
    'dist_center', 'dist_border',
    'curve_x', 'curve_y', 'sharpness',
    'delta_ncc', 'dist_v25_v46', 'v25_ml_score', 'v25_presence_score',
    'v46_score', 'v46_consensus'
]

results = []
for f in features:
    if f not in df_eval.columns: continue
    
    val_b = df_eval[df_eval['label'] == 1][f].values
    val_c = df_eval[df_eval['label'] == 0][f].values
    
    if len(val_b) == 0 or len(val_c) == 0: continue
    
    try:
        auc = roc_auc_score(df_eval['label'], df_eval[f].fillna(0))
        if auc < 0.5:
            auc = 1.0 - auc
            direction = '-'
        else:
            direction = '+'
            
        results.append({
            'Feature': f,
            'AUC': auc,
            'Dir': direction,
            'Med_B': np.nanmedian(val_b),
            'Med_C': np.nanmedian(val_c)
        })
    except:
        pass
        
res_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("=== V47 FORENSIC FEATURE ANALYSIS ===")
print("Separating Population B (True Rescues) vs Population C (Absent Background)")
print(res_df.to_string(index=False))
res_df.to_csv('phase2/V47_RESEARCH/v47_feature_analysis.csv', index=False)
