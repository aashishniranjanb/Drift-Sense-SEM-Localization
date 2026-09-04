import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import pickle

def train_ranker(df):
    # Only train on pairs where gt_found == 1
    train_df = df[df['gt_found'] == 1].copy()
    
    # Extract relative features (value - median(value in pair))
    feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                   'dist_to_center', 'neigh_cons', 'grad_ncc']
                   
    for c in feature_cols:
        train_df[c + '_rel'] = train_df.groupby('pair_id')[c].transform(lambda x: x - x.median())
        
    # We will use both absolute and relative features, plus family_ratio
    train_df['family_ratio'] = train_df.groupby('pair_id')['family_population'].transform(lambda x: x / x.count())
    
    X_cols = feature_cols + [c + '_rel' for c in feature_cols] + ['family_ratio']
    X = train_df[X_cols]
    y = train_df['label']
    
    # Train lightweight Gradient Boosting
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    model.fit(X, y)
    
    # Wrap in pipeline with feature names
    pipeline = {
        'model': model,
        'features': X_cols
    }
    with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Ranker trained. Train accuracy: {model.score(X, y):.3f}")
    return pipeline

def train_presence(df, ranker_pipeline):
    # We aggregate features for each pair_id
    # First, predict ranker scores for all candidates
    feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                   'dist_to_center', 'neigh_cons', 'grad_ncc']
    for c in feature_cols:
        df[c + '_rel'] = df.groupby('pair_id')[c].transform(lambda x: x - x.median())
    df['family_ratio'] = df.groupby('pair_id')['family_population'].transform(lambda x: x / x.count())
    
    X_cols = ranker_pipeline['features']
    df['ranker_score'] = ranker_pipeline['model'].predict_proba(df[X_cols])[:, 1]
    
    # Aggregate
    agg_rows = []
    for pair_id, group in df.groupby('pair_id'):
        group = group.sort_values('ranker_score', ascending=False)
        top1 = group.iloc[0]
        top2 = group.iloc[1] if len(group) > 1 else top1
        
        agg_rows.append({
            'pair_id': pair_id,
            'gt_found': top1['gt_found'],
            'top1_score': top1['ranker_score'],
            'top2_score': top2['ranker_score'],
            'margin': top1['ranker_score'] - top2['ranker_score'],
            'top1_corr': top1['corr_score'],
            'top1_ctx': top1['context_combined'],
            'top1_neigh': top1['neigh_cons'],
            'top1_grad': top1['grad_ncc'],
            'pitch_x': top1['pitch_x'],
            'pitch_y': top1['pitch_y'],
            'mode_strong': 1 if top1['periodicity_mode'] == 'STRONG' else 0
        })
        
    agg_df = pd.DataFrame(agg_rows)
    X_agg_cols = ['top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']
    X_agg = agg_df[X_agg_cols]
    y_agg = agg_df['gt_found']
    
    # Calibrated Logistic Regression for presence
    base_lr = LogisticRegression(class_weight='balanced', C=0.1)
    model = CalibratedClassifierCV(base_lr, cv=5, method='sigmoid')
    model.fit(X_agg, y_agg)
    
    pipeline = {
        'model': model,
        'features': X_agg_cols
    }
    with open('phase2/V25_CHAMPIONSHIP/presence.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Presence classifier trained. Train accuracy: {model.score(X_agg, y_agg):.3f}")

if __name__ == '__main__':
    df = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')
    ranker = train_ranker(df)
    train_presence(df, ranker)
