import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('phase2/V25_SCORE_MAX/pairwise_gap_features.csv')
features = [
    'delta_corr', 'delta_psr', 'delta_context128', 'delta_context_combined', 
    'delta_neigh', 'delta_gradient', 'delta_phase', 'delta_center', 'delta_family'
]

# Create symmetric negatives
df_neg = df.copy()
for f in features:
    df_neg[f] = -df_neg[f]
df_neg['label'] = 0

df_full = pd.concat([df, df_neg], ignore_index=True)

X = df_full[features]
y = df_full['label']

model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)

# OOF Evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df_full))

for train_idx, val_idx in skf.split(X, y):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

print(f"OOF AUC: {roc_auc_score(y, oof):.4f}")
print(f"OOF Accuracy: {accuracy_score(y, oof > 0.5):.4f}")

# Train on all data
model.fit(X, y)
with open('phase2/V26_CHAMPIONSHIP/pairwise_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)
print("Pairwise model saved to phase2/V26_CHAMPIONSHIP/pairwise_model.pkl")
