import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_calibrator():
    df = pd.read_csv("results/v14/presence_features.csv")
    
    # We will predict gt_found
    y = df["gt_found"].values
    
    # Features
    features = ["corr_score", "psr", "peak_margin", "context_128", "phase_residual", "ambiguity_index", "center_prior"]
    X = df[features].values
    
    # Fill any NaNs
    X = np.nan_to_num(X, nan=0.0)
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    print(f"Calibrator Training AUC: {auc:.4f}")
    
    os.makedirs("production_engine", exist_ok=True)
    with open("production_engine/score_calibrator.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
        
    print("Saved calibrator to production_engine/score_calibrator.pkl")

if __name__ == "__main__":
    train_calibrator()
