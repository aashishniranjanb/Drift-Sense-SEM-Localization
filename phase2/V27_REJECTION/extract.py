import sys
import os
import cv2
import pandas as pd
from tqdm import tqdm
from unittest.mock import MagicMock

sys.path.append('phase2')
import V25_CHAMPIONSHIP.v25_pipeline as v25

captured_features = []

original_predict_proba = v25.presence_model['model'].predict_proba

def mock_predict_proba(X):
    # X is a DataFrame. We capture it.
    # Note: we also need pair_id and gt_found, but X doesn't have it.
    # We will append to a global list and later match it.
    captured_features.append(X.iloc[0].to_dict())
    return original_predict_proba(X)

v25.presence_model['model'].predict_proba = mock_predict_proba

def extract():
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    
    # We will just run the pipeline on all pairs and capture the features.
    # Wait, run_v25_localization doesn't return the exact final candidate's raw cx, cy, 
    # it returns subpixel rx, ry.
    # That's fine, we can use the returned x, y to calculate loc_err.
    
    for idx, row in tqdm(pairs.iterrows(), total=len(pairs)):
        ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
        search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
        
        res = v25.run_v25_localization(ref, search)
        
        # The last captured feature dict is for this pair
        feat = captured_features[-1]
        feat['pair_id'] = row['pair_id']
        feat['gt_found'] = row['gt_found']
        feat['gt_x'] = row['gt_x']
        feat['gt_y'] = row['gt_y']
        feat['pred_x'] = res['x']
        feat['pred_y'] = res['y']
        
    df = pd.DataFrame(captured_features)
    df.to_csv('phase2/V27_REJECTION/v25_rejection_features.csv', index=False)
    print("Features extracted successfully.")

if __name__ == '__main__':
    extract()
