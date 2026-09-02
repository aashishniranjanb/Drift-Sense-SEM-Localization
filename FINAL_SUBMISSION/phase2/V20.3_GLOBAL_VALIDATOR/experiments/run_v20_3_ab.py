import os
import sys
import numpy as np
import pandas as pd
import cv2
from scipy.stats import entropy
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def extract_global_features(search_img, ref_img, scale, theta):
    ref_w = int(round(ref_img.shape[1] / scale))
    ref_h = int(round(ref_img.shape[0] / scale))
    if ref_w < 10 or ref_h < 10: 
        return None
    ref_resized = cv2.resize(ref_img, (ref_w, ref_h))
    
    if abs(theta) > 0.1:
        M = cv2.getRotationMatrix2D((ref_w/2, ref_h/2), theta, 1.0)
        ref_template = cv2.warpAffine(ref_resized, M, (ref_w, ref_h))
    else:
        ref_template = ref_resized
        
    corr = cv2.matchTemplate(search_img, ref_template, cv2.TM_CCOEFF_NORMED)
    
    feats = {}
    
    # 1. Entropy
    corr_pos = np.clip(corr, 0, 1)
    hist, _ = np.histogram(corr_pos.ravel(), bins=50, density=True)
    feats['corr_entropy'] = entropy(hist + 1e-9)
    
    # 2. Peaks info
    c_work = corr.copy()
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(c_work)
    
    feats['max_corr'] = max_val
    
    # Extract top peaks
    peaks = []
    for _ in range(50):
        _, val, _, loc = cv2.minMaxLoc(c_work)
        if val < 0.3: break
        peaks.append(val)
        y1, y2 = max(0, loc[1]-10), min(c_work.shape[0], loc[1]+10)
        x1, x2 = max(0, loc[0]-10), min(c_work.shape[1], loc[0]+10)
        c_work[y1:y2, x1:x2] = -1
        
    feats['num_peaks_90'] = sum([1 for p in peaks if p >= 0.9 * max_val])
    feats['peak_margin'] = max_val - peaks[1] if len(peaks) > 1 else max_val
    
    # 3. Peak Concentration
    win = 10
    h, w = corr.shape
    y1, y2 = max(0, max_loc[1]-win), min(h, max_loc[1]+win)
    x1, x2 = max(0, max_loc[0]-win), min(w, max_loc[0]+win)
    peak_sum = np.sum(corr_pos[y1:y2, x1:x2])
    total_sum = np.sum(corr_pos)
    feats['peak_concentration'] = peak_sum / (total_sum + 1e-9)
    
    # 4. Spectral energy of correlation
    f = np.fft.fft2(corr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    cy, cx = mag.shape[0]//2, mag.shape[1]//2
    center_energy = np.sum(mag[cy-5:cy+5, cx-5:cx+5])
    total_energy = np.sum(mag)
    feats['spectral_energy'] = center_energy / (total_energy + 1e-9)
    
    return feats

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'phase2_dev'))
    df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    
    all_feats = []
    print("Extracting global features...")
    for idx, row in df_pairs.iterrows():
        ref_path = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])
        
        ref_img = cv2.imread(ref_path, 0)
        search_img = cv2.imread(search_path, 0)
        if ref_img is None or search_img is None: continue
        
        scale = float(row.get("gt_scale", 10.0))
        if scale < 0.1: scale = 10.0
        theta = float(row.get("gt_theta", 0.0))
        gt_found = int(row.get("gt_found", 1))
        
        feats = extract_global_features(search_img, ref_img, scale, theta)
        if feats is not None:
            feats['pair_id'] = row['pair_id']
            feats['set_type'] = row['set_type']
            feats['label'] = gt_found
            all_feats.append(feats)
            
    df = pd.DataFrame(all_feats)
    print(f"Extracted features for {len(df)} pairs.")
    
    # fixed split
    pairs = df['pair_id'].unique()
    np.random.seed(42)
    np.random.shuffle(pairs)
    train_pairs = pairs[:int(len(pairs)*0.6)]
    val_pairs = pairs[int(len(pairs)*0.6):int(len(pairs)*0.8)]
    test_pairs = pairs[int(len(pairs)*0.8):]
    
    train_idx = df['pair_id'].isin(train_pairs)
    val_idx = df['pair_id'].isin(val_pairs)
    test_idx = df['pair_id'].isin(test_pairs)
    
    feature_cols = ['corr_entropy', 'max_corr', 'num_peaks_90', 'peak_margin', 'peak_concentration', 'spectral_energy']
    
    # V20.3-A & B Ablation
    print("V20.3-B Feature Separability (Val ROC-AUC):")
    res = "# Phase V20.3 Global Feature Separability\n\n| Feature | ROC-AUC (Val) |\n|---|---|\n"
    for col in feature_cols:
        auc = roc_auc_score(df[val_idx]['label'], df[val_idx][col])
        if auc < 0.5:
            auc = 1.0 - auc # invert if negatively correlated
        print(f" - {col}: {auc:.4f}")
        res += f"| {col} | {auc:.4f} |\n"
        
    X_train = df.loc[train_idx, feature_cols].values
    y_train = df.loc[train_idx, 'label'].values
    X_val = df.loc[val_idx, feature_cols].values
    y_val = df.loc[val_idx, 'label'].values
    X_test = df.loc[test_idx, feature_cols].values
    y_test = df.loc[test_idx, 'label'].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train_s, y_train)
    
    y_val_prob = clf.predict_proba(X_val_s)[:, 1]
    best_f1, best_t = -1, 0.5
    for t in np.linspace(0, 1, 101):
        f1 = f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            
    y_test_prob = clf.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_prob >= best_t).astype(int)
    
    f1_t = f1_score(y_test, y_test_pred, zero_division=0)
    prec_t = precision_score(y_test, y_test_pred, zero_division=0)
    rec_t = recall_score(y_test, y_test_pred, zero_division=0)
    auc_t = roc_auc_score(y_test, y_test_prob)
    
    df_test = df[test_idx].copy()
    df_test['pred'] = y_test_pred
    set_c = df_test[df_test['set_type'] == 'SetC']
    fpr_c = set_c['pred'].mean() if len(set_c) > 0 else 0
    
    res += f"\n## V20.3-C (LogReg on Global Features)\n"
    res += f"- Test AUC: {auc_t:.4f}\n"
    res += f"- Test F1: {f1_t:.4f}\n"
    res += f"- Test Precision: {prec_t:.4f}\n"
    res += f"- Test Recall: {rec_t:.4f}\n"
    res += f"- Test Set C FPR: {fpr_c:.4f} ({len(set_c)} samples)\n"
    
    print("\n--- RESULTS ---")
    print(res)
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'V20_3_RESULTS.md')), 'w') as f:
        f.write(res)

if __name__ == "__main__":
    main()
