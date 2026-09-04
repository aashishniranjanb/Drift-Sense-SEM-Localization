import os
import sys
import numpy as np
import pandas as pd
import time
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataset_generator import generate_finfet_layout, generate_pair
from phase2.inference_phase2 import perform_phase2_localization
from phase2 import rejection

from src.presence_engine import (
    v20_a_baseline,
    v20_b_multi_evidence,
    train_v20_d_classifier,
    predict_v20_d_classifier,
    v20_e_select_threshold
)

def generate_v20_datasets():
    print("Generating Set A (70 Nominal)...")
    canvas = generate_finfet_layout(10000, 10000, seed=42)
    set_a = []
    for i in range(70):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"A_{i}", "finfet", "val", 1.0, seed=42+i)
        set_a.append({"ref": ref, "search": search, "gt_found": 1, "set": "A"})
        
    print("Generating Set B (70 Degraded)...")
    set_b = []
    for i in range(70):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"B_{i}", "finfet", "val", 2.5, seed=100+i)
        set_b.append({"ref": ref, "search": search, "gt_found": 1, "set": "B"})
        
    print("Generating Set C (40 Absent - Hard Negatives)...")
    set_c = []
    rng = np.random.RandomState(200)
    for i in range(40):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"C_{i}", "finfet", "val", 1.0, seed=200+i)
        h, w = canvas.shape
        search_size = 1000
        sw_patch = int(round(search_size / 0.10))
        sh_patch = int(round(search_size / 0.10))
        
        sx1 = min(w - sw_patch, 0)
        sy1 = min(h - sh_patch, 0)
        search_raw = canvas[sy1:sy1+sh_patch, sx1:sx1+sw_patch]
        import cv2
        search_img_resized = cv2.resize(search_raw, (search_size, search_size), interpolation=cv2.INTER_AREA)
        
        from dataset_generator import apply_sem_acquisition_effects
        search_final = apply_sem_acquisition_effects(
            search_img_resized, blur_sigma=1.0, dose_lambda=80.0,
            gaussian_noise_std=0.03, edge_factor=0.12, charging_std=0.02, seed=200+i
        )
        set_c.append({"ref": ref, "search": search_final, "gt_found": 0, "set": "C"})
        
    return set_a + set_b + set_c

def main():
    datasets = generate_v20_datasets()
    np.random.seed(42)
    np.random.shuffle(datasets)
    
    results = []
    features_list = []
    
    original_extract = rejection.extract_presence_features
    
    def patched_extract(corr_plane, peak_x, peak_y, rotated_template, search_img, context_score=0.0, phase_residual=0.0):
        feats = original_extract(corr_plane, peak_x, peak_y, rotated_template, search_img, context_score, phase_residual)
        features_list.append(feats)
        return feats
    
    print("Running pairs through inference engine to extract features...")
    with patch("phase2.inference_phase2.extract_presence_features", side_effect=patched_extract):
        for idx, item in enumerate(datasets):
            t0 = time.perf_counter()
            out = perform_phase2_localization(item["ref"], item["search"])
            dt = time.perf_counter() - t0
            
            feat = features_list[-1]
            results.append({
                "idx": idx,
                "set": item["set"],
                "gt_found": item["gt_found"],
                "baseline_found": out["found"],
                "baseline_score": out["score"],
                "latency_ms": dt * 1000.0,
                **feat
            })
            
    df = pd.DataFrame(results)
    
    feature_cols = ["max_score", "delta_s", "psr", "context_score", "phase_residual"]
    X = df[feature_cols].values
    y = df["gt_found"].values
    
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    train_idx = indices[:90]
    test_idx = indices[90:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    df["v20_b_score"] = df.apply(lambda row: v20_b_multi_evidence(row), axis=1)
    
    clf = train_v20_d_classifier(X_train, y_train)
    df["v20_d_prob"] = predict_v20_d_classifier(clf, X)
    
    opt_thresh = v20_e_select_threshold(df.iloc[train_idx]["v20_d_prob"].values, y_train)
    df["v20_e_pred"] = (df["v20_d_prob"] >= opt_thresh).astype(int)
    
    df_test = df.iloc[test_idx]
    y_true_test = df_test["gt_found"]
    y_pred_test = df_test["v20_e_pred"]
    y_prob_test = df_test["v20_d_prob"]
    
    cm = confusion_matrix(y_true_test, y_pred_test)
    prec = precision_score(y_true_test, y_pred_test, zero_division=0)
    rec = recall_score(y_true_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_true_test, y_prob_test)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_test, y_prob_test)
    pr_auc = auc(rec_curve, prec_curve)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")), exist_ok=True)
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "v20_metrics_report.md"))
    
    with open(report_path, "w") as f:
        f.write("# Phase V20 Metrics Report\n\n")
        f.write("## Overall Test Set Metrics (V20-D & V20-E)\n")
        f.write(f"- **Total Test Cases**: {len(df_test)}\n")
        f.write(f"- **Precision**: {prec:.4f}\n")
        f.write(f"- **Recall**: {rec:.4f}\n")
        f.write(f"- **F1 Score**: {f1:.4f}\n")
        f.write(f"- **ROC-AUC**: {roc_auc:.4f}\n")
        f.write(f"- **PR-AUC**: {pr_auc:.4f}\n")
        f.write(f"- **FPR**: {fpr:.4f}\n")
        f.write(f"- **FNR**: {fnr:.4f}\n")
        f.write(f"- **Chosen Threshold (from train)**: {opt_thresh:.4f}\n")
        f.write(f"- **Median Runtime per Pair**: {df["latency_ms"].median():.2f} ms\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write("| | Predicted Absent | Predicted Present |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **True Absent** | {tn} | {fp} |\n")
        f.write(f"| **True Present** | {fn} | {tp} |\n\n")
        
        f.write("## Segmented Metrics (Test Set)\n")
        for s in ["A", "B", "C"]:
            subset = df_test[df_test["set"] == s]
            if len(subset) == 0: continue
            y_sub = subset["gt_found"]
            p_sub = subset["v20_e_pred"]
            acc = (y_sub == p_sub).mean()
            f.write(f"- **Set {s} Accuracy**: {acc:.4f} ({len(subset)} cases)\n")
            
    df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "v20_predictions.csv")), index=False)
    print(f"Done! Report saved to {report_path}")

if __name__ == "__main__":
    main()


