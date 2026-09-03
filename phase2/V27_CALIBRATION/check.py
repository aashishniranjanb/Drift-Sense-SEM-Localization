import pandas as pd
from scipy.stats import spearmanr
import numpy as np

df = pd.read_csv('phase2/V27_REJECTION/oof_predictions.csv')
v25_preds = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
df = pd.merge(df, v25_preds[['pair_id', 'x', 'y']], on='pair_id')
pairs = pd.read_csv('data/phase2_dev/pairs.csv')
merged = pd.merge(pairs, df, on='pair_id', suffixes=('', '_df'))

merged['loc_err'] = np.hypot(merged['x'] - merged['gt_x'], merged['y'] - merged['gt_y'])

# Let's use the V25 logic for correctness to compare apples to apples
correctness = []
for _, row in merged.iterrows():
    gt = row['gt_found']
    pr = row['gt_found'] # Assuming we accepted everything correctly for a moment... no wait.
    # In benchmark_phase2, it uses the PREDICTED found to determine the confusion matrix bin!
    pass

# Wait! The official spearman calculation uses ALL PREDICTIONS, 
# and sets correctness = 1 IF failure_mode in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"]
correctness = []
for _, row in merged.iterrows():
    gt = row['gt_found']
    pr = 1 if row['oof_combined'] > 0.68 else 0  # Our optimal OOF threshold
    
    if gt == 1 and pr == 1:
        if row['loc_err'] <= 5.0: correctness.append(1)
        else: correctness.append(0)
    elif gt == 0 and pr == 0:
        correctness.append(1)
    else:
        correctness.append(0)

s1, _ = spearmanr(merged['oof_combined'], correctness)
s2, _ = spearmanr(merged['top1_score'], correctness)

print(f"OOF Combined Spearman: {s1:.4f}")
print(f"Top1 Score Spearman: {s2:.4f}")
