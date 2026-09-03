import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

def train_new_presence():
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    df = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')
    
    with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'rb') as f:
        ranker = pickle.load(f)
        
    feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                   'dist_to_center', 'neigh_cons', 'grad_ncc']
    for c in feature_cols:
        df[c + '_rel'] = df.groupby('pair_id')[c].transform(lambda x: x - x.median())
    df['family_ratio'] = df.groupby('pair_id')['family_population'].transform(lambda x: x / x.count())
    
    X_cols = ranker['features']
    df['ranker_score'] = ranker['model'].predict_proba(df[X_cols])[:, 1]
    
    agg_rows = []
    for pair_id, group in df.groupby('pair_id'):
        group = group.sort_values('ranker_score', ascending=False)
        scores = group['ranker_score'].values
        
        top1 = group.iloc[0]
        top1_score = scores[0]
        top2_score = scores[1] if len(scores) > 1 else top1_score
        top5_score = scores[4] if len(scores) > 4 else scores[-1]
        
        pair_info = pairs[pairs['pair_id'] == pair_id].iloc[0]
        pair_present = 1 if pair_info['set_type'] in ['SetA', 'SetB'] else 0
        
        agg_rows.append({
            'pair_id': pair_id,
            'set_type': pair_info['set_type'],
            'actual_present': pair_present,
            'top1_score': top1_score,
            'top2_score': top2_score,
            'top5_score': top5_score,
            'gap_1_2': top1_score - top2_score,
            'gap_1_5': top1_score - top5_score,
            'top1_corr': top1['corr_score'],
            'top1_ctx': top1['context_combined'],
            'top1_neigh': top1['neigh_cons'],
            'top1_grad': top1['grad_ncc'],
            'top1_phase_pen': top1['phase_penalty'],
            'mode_strong': 1 if top1['periodicity_mode'] == 'STRONG' else 0,
        })
        
    agg_df = pd.DataFrame(agg_rows)
    
    features = ['top1_score', 'gap_1_2', 'gap_1_5',
                'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'top1_phase_pen', 'mode_strong']
    X = agg_df[features]
    y = agg_df['actual_present']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        m = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=200, max_depth=5, l2_regularization=0.5, random_state=42), cv=3, method='sigmoid')
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        
    print("OOF Predictions stats:")
    print("Present cases min/mean/max:", oof_preds[y==1].min(), oof_preds[y==1].mean(), oof_preds[y==1].max())
    print("Absent cases min/mean/max:", oof_preds[y==0].min(), oof_preds[y==0].mean(), oof_preds[y==0].max())
        
if __name__ == '__main__':
    train_new_presence()
