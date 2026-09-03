import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import json

df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')

# Step 1: P(candidate correct) training (Experiment C part 2, or just use it as part of rejection)
# Actually, the user says:
# "candidate_confidence = P(present) * P(candidate_correct)"
# Let's train a 'Candidate Correct' model on present pairs
present_df = df[df['gt_found'] == 1].copy()

# Sort by v25_ml_score per pair and get top-1
top1_df = present_df.sort_values(['pair_id', 'v25_ml_score'], ascending=[True, False]).groupby('pair_id').head(1)

# P(candidate correct) model features
correct_features = ['corr_score', 'context_combined', 'neigh_cons', 'grad_ncc', 'v25_ml_score', 'mode_strong']
X_corr = top1_df[correct_features]
y_corr = top1_df['is_correct']

clf_corr = RandomForestClassifier(max_depth=4, random_state=42)
clf_corr.fit(X_corr, y_corr)

# Step 2: P(present) training
# Get Top-1 candidate for ALL pairs (present and absent)
all_top1_df = df.sort_values(['pair_id', 'v25_ml_score'], ascending=[True, False]).groupby('pair_id').head(1).copy()
all_top1_df['top1_margin'] = all_top1_df['v25_ml_score'] - df.sort_values(['pair_id', 'v25_ml_score'], ascending=[True, False]).groupby('pair_id').nth(1)['v25_ml_score'].values

# P(present) features
pres_features = ['v25_ml_score', 'top1_margin', 'corr_score', 'context_combined', 'neigh_cons', 'grad_ncc', 'mode_strong']
X_pres = all_top1_df[pres_features]
y_pres = all_top1_df['gt_found']

clf_pres = HistGradientBoostingClassifier(max_depth=3, random_state=42)
clf_pres.fit(X_pres, y_pres)

# Save models
with open('phase2/V26_CHAMPIONSHIP/rejection_models.pkl', 'wb') as f:
    pickle.dump({
        'clf_corr': clf_corr,
        'corr_features': correct_features,
        'clf_pres': clf_pres,
        'pres_features': pres_features
    }, f)

print("Rejection models trained and saved.")
