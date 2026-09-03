import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('phase2/V47_RESEARCH/v47_candidate_cache/features.csv')

def get_target(r):
    if r['gt_found'] == 0: return 0
    if np.isnan(r['gt_x']) or np.isnan(r['gt_y']): return 0
    d = np.hypot(r['v46_cx'] - r['gt_x'], r['v46_cy'] - r['gt_y'])
    if d <= 5.0: return 1
    return 0

df['is_correct'] = df.apply(get_target, axis=1)

features = [
    'ncc', 'grad', 'ctx', 'phase',
    'ncc_pct', 'grad_pct',
    'prom5_ncc', 'prom10_ncc', 'prom20_ncc',
    'z5_ncc', 'z10_ncc',
    'comp10', 'comp20', 'd1', 'd2',
    'dist_center', 'dist_border',
    'sharpness', 'delta_ncc', 'dist_v25_v46'
]

X = df[features].fillna(0).values
y = df['is_correct'].values

hgb2 = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, random_state=42)
hgb2.fit(X, y)

with open('phase2/V47_RESEARCH/v47_hgb2.pkl', 'wb') as f:
    pickle.dump({'model': hgb2, 'features': features}, f)
print("Saved final V47 validator model")
