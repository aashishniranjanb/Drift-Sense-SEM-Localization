import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def train_new_presence():
    df = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')
    
    # Load ranker
    with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'rb') as f:
        ranker = pickle.load(f)
        
    # Recompute relative features for ranker
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
        top1 = group.iloc[0]
        top2 = group.iloc[1] if len(group) > 1 else top1
        top5 = group.iloc[4] if len(group) > 4 else group.iloc[-1]
        
        agg_rows.append({
            'pair_id': pair_id,
            'gt_found': top1['gt_found'],
            'top1_score': top1['ranker_score'],
            'margin_1_2': top1['ranker_score'] - top2['ranker_score'],
            'margin_1_5': top1['ranker_score'] - top5['ranker_score'],
            'top1_corr': top1['corr_score'],
            'top1_ctx': top1['context_combined'],
            'top1_neigh': top1['neigh_cons'],
            'top1_grad': top1['grad_ncc'],
            'top1_phase_pen': top1['phase_penalty'],
            'mode_strong': 1 if top1['periodicity_mode'] == 'STRONG' else 0,
            'pitch_x': top1['pitch_x'],
            'pitch_y': top1['pitch_y']
        })
        
    agg_df = pd.DataFrame(agg_rows)
    features = ['top1_score', 'margin_1_2', 'margin_1_5', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'top1_phase_pen', 'mode_strong']
    X = agg_df[features]
    y = agg_df['gt_found']
    
    # Train HGBC
    base_model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, l2_regularization=1.0, random_state=42)
    calibrated = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
    calibrated.fit(X, y)
    
    # Get OOF predictions for threshold tuning
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        m = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=100, max_depth=5, l2_regularization=1.0, random_state=42), cv=3, method='sigmoid')
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
        
    best_thresh = 0.5
    best_f1 = 0
    # Search threshold strictly on absent vs present
    for t in np.linspace(0.1, 0.9, 100):
        pred_absent = (oof_preds < t)
        true_absent = (y == 0)
        tp = np.sum(pred_absent & true_absent)
        fp = np.sum(pred_absent & ~true_absent)
        fn = np.sum(~pred_absent & true_absent)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Best OOF Rejection F1: {best_f1:.4f} at threshold {best_thresh:.3f}")
    
    pipeline = {
        'model': calibrated,
        'features': features,
        'threshold': best_thresh
    }
    with open('phase2/V25_FINAL_BONUS/presence_hgbc.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
if __name__ == '__main__':
    train_new_presence()
