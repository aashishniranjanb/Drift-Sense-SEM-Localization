import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import os
import joblib

def run_experiment():
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    
    full_pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    # For splitting, we only care about the pair IDs present in full_pairs.
    # We will use indexing based on rows in full_pairs (0-179)
    train_pairs = set(full_pairs.loc[0:107, 'pair_id'])
    val_pairs = set(full_pairs.loc[108:143, 'pair_id'])
    test_pairs = set(full_pairs.loc[144:179, 'pair_id'])
    
    train_mask = df['pair_id'].isin(train_pairs)
    val_mask = df['pair_id'].isin(val_pairs)
    test_mask = df['pair_id'].isin(test_pairs)
    
    features = [
        'corr_score', 'psr', 'context_128', 'phase_residual', 'phase_penalty', 'dist_to_center', 'family_population', 'peak_margin',
        'ncc_delta', 'center_delta', 'phase_delta', 'context_delta', 'rank_by_ncc', 'rank_by_phase', 'rank_by_center', 'family_size', 'nearest_competitor_ncc'
    ]
    
    X_train = df[train_mask][features].fillna(0)
    y_train = df[train_mask]['is_correct']
    
    X_val = df[val_mask][features].fillna(0)
    y_val = df[val_mask]['is_correct']
    
    # P0: V18-C
    def p0_score(row):
        fam_pop = row.get("family_population", 1)
        w_center = 0.12 if fam_pop > 3 else 0.04
        center_penalty = (row["dist_to_center"] / 250.0) ** 2
        return row["corr_score"] + 0.15 * row["context_128"] - 0.20 * row["phase_penalty"] - w_center * center_penalty
        
    def evaluate(df_subset, score_col):
        correct_top1 = 0
        total_pairs_with_gt = 0
        
        for pid, group in df_subset.groupby('pair_id'):
            has_gt = group['is_correct'].sum() > 0
            if has_gt:
                total_pairs_with_gt += 1
                best_idx = group[score_col].idxmax()
                if group.loc[best_idx, 'is_correct'] == 1:
                    correct_top1 += 1
                    
        return correct_top1 / total_pairs_with_gt if total_pairs_with_gt > 0 else 0
        
    df['P0_score'] = df.apply(p0_score, axis=1)
    
    p0_acc = evaluate(df[val_mask], 'P0_score')
    print(f"P0 (V18-C) conditional Top-1 on VAL: {p0_acc:.4f}")
    
    p1 = LogisticRegression(class_weight='balanced', max_iter=1000)
    p1.fit(X_train, y_train)
    df.loc[val_mask, 'P1_score'] = p1.predict_proba(X_val)[:, 1]
    p1_acc = evaluate(df[val_mask], 'P1_score')
    print(f"P1 (LogReg) conditional Top-1 on VAL: {p1_acc:.4f}")
    
    p2 = HistGradientBoostingClassifier(class_weight='balanced')
    p2.fit(X_train, y_train)
    df.loc[val_mask, 'P2_score'] = p2.predict_proba(X_val)[:, 1]
    p2_acc = evaluate(df[val_mask], 'P2_score')
    print(f"P2 (HGBC) conditional Top-1 on VAL: {p2_acc:.4f}")
    
    # Save the best model
    best_acc = max(p1_acc, p2_acc)
    best_model_name = "P1" if p1_acc >= p2_acc and p1_acc >= 0.65 else ("P2" if p2_acc >= 0.65 else "P0")
    if p1_acc >= 0.65 and p2_acc >= 0.65:
        # KEEP P1 if same but simpler preferred (so if P1 is close to P2, pick P1, actually if P1 meets threshold prefer it)
        # But wait, instruction: "KEEP P1 if same but simpler preferred"
        if p2_acc - p1_acc > 0.02:
            best_model_name = "P2"
        else:
            best_model_name = "P1"
            
    print(f"Chosen model for next step based on VAL: {best_model_name}")
    
    # Predict for TEST set to save
    if best_model_name == "P1":
        df.loc[test_mask, 'final_score'] = p1.predict_proba(df[test_mask][features].fillna(0))[:, 1]
        joblib.dump(p1, "phase2/V22_CHAMPIONSHIP/results/best_ranker.pkl")
    elif best_model_name == "P2":
        df.loc[test_mask, 'final_score'] = p2.predict_proba(df[test_mask][features].fillna(0))[:, 1]
        joblib.dump(p2, "phase2/V22_CHAMPIONSHIP/results/best_ranker.pkl")
    else:
        df.loc[test_mask, 'final_score'] = df.loc[test_mask, 'P0_score']
        
    df.to_csv("phase2/V22_CHAMPIONSHIP/results/blast2_ranking_results.csv", index=False)
    
    # Also write a small text file passing info to the next script
    with open("phase2/V22_CHAMPIONSHIP/results/chosen_model.txt", "w") as f:
        f.write(best_model_name)

if __name__ == "__main__":
    run_experiment()
