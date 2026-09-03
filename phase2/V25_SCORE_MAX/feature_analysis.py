import pandas as pd
import numpy as np

pairs = pd.read_csv('data/phase2_dev/pairs.csv')
df = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')

import pickle
with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'rb') as f:
    ranker = pickle.load(f)

X_cols = ranker['features']
df['ranker_score'] = ranker['model'].predict_proba(df[X_cols])[:, 1]

agg_rows = []
for pair_id, group in df.groupby('pair_id'):
    group = group.sort_values('ranker_score', ascending=False)
    scores = group['ranker_score'].values
    top1 = group.iloc[0]
    pair_info = pairs[pairs['pair_id'] == pair_id].iloc[0]
    agg_rows.append({
        'pair_id': pair_id,
        'is_present': 1 if pair_info['set_type'] in ['SetA', 'SetB'] else 0,
        'top1_score': scores[0],
        'gap_1_2': scores[0] - (scores[1] if len(scores) > 1 else scores[0]),
        'top1_neigh': top1['neigh_cons'],
        'top1_grad': top1['grad_ncc'],
        'top1_corr': top1['corr_score'],
    })

df_agg = pd.DataFrame(agg_rows)
print("Correlation of features with is_present:")
print(df_agg.corr()['is_present'].sort_values())
