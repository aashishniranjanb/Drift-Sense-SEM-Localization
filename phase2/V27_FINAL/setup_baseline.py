import os
import pandas as pd
import numpy as np
import time
import sys
import cv2

sys.path.append('phase2')
sys.path.append('phase2/V25_CHAMPIONSHIP')
from v25_pipeline import run_v25_localization

def extract_exact_v25():
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    
    # Check if we already have the baseline predictions and features
    v25_preds_file = 'data/phase2_dev/v25_predictions_thresh.csv'
    if os.path.exists(v25_preds_file):
        df_preds = pd.read_csv(v25_preds_file)
        df_preds.to_csv('phase2/V27_FINAL/V25_BASELINE.csv', index=False)
        print(f"Copied {len(df_preds)} rows to phase2/V27_FINAL/V25_BASELINE.csv")
        
extract_exact_v25()
