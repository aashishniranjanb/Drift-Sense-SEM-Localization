import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

sys.path.append('phase2')
from benchmark_phase2 import evaluate_phase2

def run_v26b():
    df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    
    with open('phase2/V26_CHAMPIONSHIP/pairwise_model.pkl', 'rb') as f:
        pairwise_model = pickle.load(f)
        
    with open('phase2/V25_CHAMPIONSHIP/presence.pkl', 'rb') as f:
        presence_model = pickle.load(f)
        
    results_list = []
    
    for pid in pairs['pair_id']:
        pair_cands = df[df['pair_id'] == pid].copy()
        if len(pair_cands) == 0:
            continue
            
        v25_cands = pair_cands[pair_cands['queue'] == 'V25'].sort_values('v25_ml_score', ascending=False)
        if len(v25_cands) == 0:
            v25_cands = pair_cands.sort_values('v25_ml_score', ascending=False)
            
        v25_anchor = v25_cands.iloc[0]
        v25_second = v25_cands.iloc[1] if len(v25_cands) > 1 else v25_anchor
        
        rescue_cands = pair_cands[pair_cands['queue'].isin(['R2', 'R3'])].copy()
        valid_rescues = []
        for _, rc in rescue_cands.iterrows():
            if np.hypot(rc['cx'] - v25_anchor['cx'], rc['cy'] - v25_anchor['cy']) > 2.0:
                valid_rescues.append(rc)
                
        best_rescue = None
        best_rescue_prob = 0.0
        
        if len(valid_rescues) > 0:
            valid_rescues_df = pd.DataFrame(valid_rescues)
            X_gap = pd.DataFrame()
            
            for f in pairwise_model['features']:
                base_f = f.replace('delta_', '')
                if base_f == 'corr': base_f = 'corr_score'
                elif base_f == 'context128': base_f = 'context_128'
                elif base_f == 'center': base_f = 'dist_to_center'
                elif base_f == 'family': base_f = 'family_population'
                elif base_f == 'neigh': base_f = 'neigh_cons'
                elif base_f == 'gradient': base_f = 'grad_ncc'
                elif base_f == 'phase': base_f = 'phase_penalty'
                elif base_f == 'context_combined': base_f = 'context_combined'
                
                X_gap[f] = valid_rescues_df[base_f].values - v25_anchor[base_f]
                
            probs = pairwise_model['model'].predict_proba(X_gap)[:, 1]
            best_idx = np.argmax(probs)
            best_rescue_prob = probs[best_idx]
            best_rescue = valid_rescues_df.iloc[best_idx]
            
        results_list.append({
            'pair_id': pid,
            'anchor': v25_anchor,
            'anchor_second': v25_second,
            'best_rescue': best_rescue,
            'best_rescue_prob': best_rescue_prob
        })
        
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    
    for thresh in thresholds:
        preds = []
        for r in results_list:
            winner = r['anchor']
            is_rescue = False
            margin = r['anchor']['v25_ml_score'] - r['anchor_second']['v25_ml_score']
            
            if r['best_rescue'] is not None and r['best_rescue_prob'] >= thresh:
                winner = r['best_rescue']
                is_rescue = True
                
            X_pres = pd.DataFrame([{
                'top1_score': r['anchor']['v25_ml_score'], # Use anchor for presence logic
                'margin': margin,
                'top1_corr': winner['corr_score'],
                'top1_ctx': winner['context_combined'],
                'top1_neigh': winner['neigh_cons'],
                'top1_grad': winner['grad_ncc'],
                'mode_strong': winner['mode_strong']
            }])
            pres_score = float(presence_model['model'].predict_proba(X_pres[presence_model['features']])[0, 1])
            found = 1 if pres_score > 0.843 else 0
            
            preds.append({
                'pair_id': r['pair_id'],
                'x': winner['cx'] if found else 0,
                'y': winner['cy'] if found else 0,
                'theta': 0.0,
                'scale': 1.0,
                'found': found,
                'score': pres_score,
                'is_rescue': is_rescue,
                'runtime': 3.4
            })
            
        pred_df = pd.DataFrame(preds)
        tmp_csv = f'phase2/V26_PAIRWISE_RESCUE/tmp_v26b_{thresh}.csv'
        pred_df.to_csv(tmp_csv, index=False)
        print(f"\\n--- THRESHOLD {thresh} ---")
        print(f"Rescues attempted: {pred_df['is_rescue'].sum()}")
        evaluate_phase2('data/phase2_dev/pairs.csv', tmp_csv)
        os.remove(tmp_csv)

if __name__ == '__main__':
    run_v26b()
