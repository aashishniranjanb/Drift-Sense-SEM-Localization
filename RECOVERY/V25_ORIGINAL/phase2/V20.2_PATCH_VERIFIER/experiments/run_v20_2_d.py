import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase2.inference_phase2 import perform_phase2_localization
from phase2 import candidate_ranker

class PatchFeatureDataset(Dataset):
    def __init__(self, data, scaler=None):
        self.data = data
        self.scaler = scaler
        
        # Extract features
        feats = []
        for row in data:
            feats.append([row['peak_margin'], row['dist_inv'], row['context']])
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.feats_scaled = self.scaler.fit_transform(feats)
        else:
            self.feats_scaled = self.scaler.transform(feats)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        ref = torch.tensor(row['ref'], dtype=torch.float32).unsqueeze(0) / 255.0
        search = torch.tensor(row['search'], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(row['label'], dtype=torch.float32)
        feat = torch.tensor(self.feats_scaled[idx], dtype=torch.float32)
        return ref, search, feat, label

class HybridTwoStreamCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 64*3 for CNN features + 3 for handcrafted features
        self.fc = nn.Sequential(
            nn.Linear(64*3 + 3, 64), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x1, x2, feats):
        f1 = self.features(x1).view(x1.size(0), -1)
        f2 = self.features(x2).view(x2.size(0), -1)
        combined_cnn = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        combined_all = torch.cat([combined_cnn, feats], dim=1)
        return self.fc(combined_all).squeeze(1)

def extract_patch(img, cx, cy, size=64):
    h, w = img.shape
    x1 = max(0, int(cx - size//2))
    y1 = max(0, int(cy - size//2))
    x2 = min(w, int(cx + size//2))
    y2 = min(h, int(cy + size//2))
    patch = img[y1:y2, x1:x2]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size))
    return patch

def augment_patch(patch):
    # small sub-pixel shift equivalent by rolling
    shift_x = np.random.randint(-2, 3)
    shift_y = np.random.randint(-2, 3)
    aug = np.roll(patch, shift_x, axis=1)
    aug = np.roll(aug, shift_y, axis=0)
    # add small noise
    noise = np.random.normal(0, 5, aug.shape).astype(np.float32)
    aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'phase2_dev'))
    df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    
    print("Running inference to extract rich candidates...")
    all_patches = []
    
    original_rank_candidates = candidate_ranker.rank_candidates
    def patched_rank_candidates(candidates):
        ranked = original_rank_candidates(candidates)
        if hasattr(patched_rank_candidates, "current_context"):
            row = patched_rank_candidates.current_context["row"]
            ref_img = patched_rank_candidates.current_context["ref_img"]
            search_img = patched_rank_candidates.current_context["search_img"]
            
            gt_found = int(row['gt_found'])
            gt_x = float(row.get('gt_x', 0))
            gt_y = float(row.get('gt_y', 0))
            
            scale = float(row.get("gt_scale", 10.0))
            if scale < 0.1: scale = 10.0
            
            ref_w = int(round(ref_img.shape[1] / scale))
            ref_h = int(round(ref_img.shape[0] / scale))
            if ref_w >= 10 and ref_h >= 10:
                ref_resized = cv2.resize(ref_img, (ref_w, ref_h))
                ref_patch = cv2.resize(ref_resized, (64, 64))
                
                pos_extracted = False
                
                # Check top 10 candidates
                for c in ranked[:10]:
                    cx, cy = c['cx'], c['cy']
                    cand_patch = extract_patch(search_img, cx, cy, size=64)
                    
                    dist_raw = c.get('nearest_cut_dist', 10.0)
                    dist_inv = 1.0 / (1.0 + dist_raw / 20.0)
                    
                    if gt_found == 1:
                        err = np.hypot(cx - gt_x, cy - gt_y)
                        if err <= 10.0:
                            label = 1
                            pos_extracted = True
                        elif err > 30.0:
                            label = 0
                        else:
                            continue # ignore borderline
                    else:
                        label = 0
                        
                    all_patches.append({
                        'pair_id': row['pair_id'],
                        'set_type': row['set_type'],
                        'ref': ref_patch,
                        'search': cand_patch,
                        'label': label,
                        'peak_margin': c.get('peak_margin', 0.0),
                        'dist_inv': dist_inv,
                        'context': c.get('context_128', 0.0)
                    })
                    
                    if label == 1:
                        # Data augmentation for positives
                        for _ in range(3):
                            aug_patch = augment_patch(cand_patch)
                            all_patches.append({
                                'pair_id': row['pair_id'],
                                'set_type': row['set_type'],
                                'ref': ref_patch,
                                'search': aug_patch,
                                'label': 1,
                                'peak_margin': c.get('peak_margin', 0.0),
                                'dist_inv': dist_inv,
                                'context': c.get('context_128', 0.0)
                            })
                            
                # If positive wasn't in top 10, extract it manually with dummy features (or just skip. Better to have it)
                if gt_found == 1 and not pos_extracted:
                    cand_patch = extract_patch(search_img, gt_x, gt_y, size=64)
                    all_patches.append({
                        'pair_id': row['pair_id'],
                        'set_type': row['set_type'],
                        'ref': ref_patch,
                        'search': cand_patch,
                        'label': 1,
                        'peak_margin': 0.05, # pseudo default for true pos
                        'dist_inv': 0.8,
                        'context': 0.8
                    })
                    for _ in range(3):
                        aug_patch = augment_patch(cand_patch)
                        all_patches.append({
                            'pair_id': row['pair_id'],
                            'set_type': row['set_type'],
                            'ref': ref_patch,
                            'search': aug_patch,
                            'label': 1,
                            'peak_margin': 0.05,
                            'dist_inv': 0.8,
                            'context': 0.8
                        })
        return ranked

    with patch("phase2.inference_phase2.rank_candidates", side_effect=patched_rank_candidates):
        for idx, row in df_pairs.iterrows():
            if idx % 10 == 0:
                print(f"Processing pair {idx}...")
            ref_path = os.path.join(data_dir, row["reference_path"])
            search_path = os.path.join(data_dir, row["search_path"])
            
            ref_img = cv2.imread(ref_path, 0)
            search_img = cv2.imread(search_path, 0)
            
            if ref_img is None or search_img is None: continue
            
            patched_rank_candidates.current_context = {"row": row, "ref_img": ref_img, "search_img": search_img}
            perform_phase2_localization(ref_img, search_img)

    df_p = pd.DataFrame(all_patches)
    print(f"Total patches: {len(df_p)}")
    
    pairs = df_p['pair_id'].unique()
    np.random.seed(42)
    np.random.shuffle(pairs)
    train_pairs = pairs[:int(len(pairs)*0.6)]
    val_pairs = pairs[int(len(pairs)*0.6):int(len(pairs)*0.8)]
    test_pairs = pairs[int(len(pairs)*0.8):]
    
    train_data = df_p[df_p['pair_id'].isin(train_pairs)].to_dict('records')
    val_data = df_p[df_p['pair_id'].isin(val_pairs)].to_dict('records')
    test_data = df_p[df_p['pair_id'].isin(test_pairs)].to_dict('records')
    
    num_pos = sum([1 for x in train_data if x['label'] == 1])
    num_neg = sum([1 for x in train_data if x['label'] == 0])
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32)
    
    print(f"Train Positive: {num_pos}, Negative: {num_neg}, pos_weight: {pos_weight.item():.2f}")
    
    ds_train = PatchFeatureDataset(train_data)
    ds_val = PatchFeatureDataset(val_data, scaler=ds_train.scaler)
    ds_test = PatchFeatureDataset(test_data, scaler=ds_train.scaler)
    
    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=32)
    test_loader = DataLoader(ds_test, batch_size=32)
    
    model = HybridTwoStreamCNN()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Hybrid V20.2-D...")
    for ep in range(15):
        model.train()
        for ref, search, feat, label in train_loader:
            optimizer.zero_grad()
            out = model(ref, search, feat)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
    def get_preds(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ref, search, feat, label in loader:
                out = model(ref, search, feat)
                probs = torch.sigmoid(out)
                preds.extend(probs.numpy())
                trues.extend(label.numpy())
        return np.array(trues), np.array(preds)
        
    y_val_true, y_val_prob = get_preds(model, val_loader)
    best_f1 = -1
    best_t = 0.5
    for t in np.linspace(0, 1, 101):
        y_val_pred = (y_val_prob >= t).astype(int)
        f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    y_test_true, y_test_prob = get_preds(model, test_loader)
    
    y_test_pred = (y_test_prob >= best_t).astype(int)
    f1_test = f1_score(y_test_true, y_test_pred, zero_division=0)
    prec_test = precision_score(y_test_true, y_test_pred, zero_division=0)
    rec_test = recall_score(y_test_true, y_test_pred, zero_division=0)
    auc_test = roc_auc_score(y_test_true, y_test_prob)
    
    test_df = df_p[df_p['pair_id'].isin(test_pairs)].copy()
    test_df['pred'] = y_test_pred
    
    # We want to measure FPR on Set C properly. For each pair, if any candidate is predicted 1, it's a FP for that pair?
    # No, the FPR on Set C candidates
    set_c = test_df[test_df['set_type'] == 'SetC']
    if len(set_c) > 0:
        fpr_c = set_c['pred'].mean()
    else:
        fpr_c = 0.0
        
    res = f"# V20.2-D Calibrated Results (CNN + Handcrafted)\n\n"
    res += f"- **Threshold tuned on Val**: {best_t:.4f} (Val F1: {best_f1:.4f})\n"
    res += f"- **Test AUC**: {auc_test:.4f}\n"
    res += f"- **Test F1**: {f1_test:.4f}\n"
    res += f"- **Test Precision**: {prec_test:.4f}\n"
    res += f"- **Test Recall**: {rec_test:.4f}\n"
    res += f"- **Test Set C FPR**: {fpr_c:.4f} ({len(set_c)} candidates)\n"
    
    print(res)
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'V20_2_D_RESULTS.md')), 'w') as f:
        f.write(res)

if __name__ == "__main__":
    main()
