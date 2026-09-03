import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def train_oof():
    df = pd.read_csv('phase2/V27_REJECTION/v25_rejection_features.csv')
    
    # Target 1: Is Present?
    y_present = df['gt_found'].values
    
    # Target 2: Is Correct? (Only on present pairs)
    df['loc_err'] = np.hypot(df['pred_x'] - df['gt_x'], df['pred_y'] - df['gt_y'])
    # Strict localization tier: we can use <=5px as "correct"
    df['is_correct'] = (df['loc_err'] <= 5.0).astype(int)
    
    features = ['top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_pres_hgb = np.zeros(len(df))
    oof_corr_hgb = np.zeros(len(df))
    
    # Train HGB models
    for train_idx, val_idx in skf.split(df, y_present):
        X_train, X_val = df.iloc[train_idx][features], df.iloc[val_idx][features]
        y_train_pres, y_val_pres = y_present[train_idx], y_present[val_idx]
        
        clf_pres = HistGradientBoostingClassifier(max_depth=3, random_state=42)
        clf_pres.fit(X_train, y_train_pres)
        oof_pres_hgb[val_idx] = clf_pres.predict_proba(X_val)[:, 1]
        
        # Correctness model: train only on present pairs
        train_present_mask = (y_train_pres == 1)
        if train_present_mask.sum() > 0:
            clf_corr = HistGradientBoostingClassifier(max_depth=3, random_state=42)
            clf_corr.fit(X_train[train_present_mask], df.iloc[train_idx]['is_correct'][train_present_mask])
            oof_corr_hgb[val_idx] = clf_corr.predict_proba(X_val)[:, 1]
            
    df['oof_pres'] = oof_pres_hgb
    df['oof_corr'] = oof_corr_hgb
    df['oof_combined'] = oof_pres_hgb * oof_corr_hgb
    
    df.to_csv('phase2/V27_REJECTION/oof_predictions.csv', index=False)
    print("OOF predictions generated and saved to phase2/V27_REJECTION/oof_predictions.csv")

if __name__ == '__main__':
    train_oof()
